use std::{
    collections::HashMap,
    fs::File,
    io::{BufWriter, Write},
    sync::atomic::{AtomicBool, Ordering},
    thread,
};

use alsa::pcm::Format;
use audio::CaptureDevice;
use circular_buffer::CircularBuffer;
use clap::{Parser, Subcommand, command};
use ndarray::Array2;
use ort::{
    inputs,
    value::{DynMapValueType, Sequence, TensorRef},
};
use signal_hook::{consts::SIGINT, iterator::Signals};

mod audio;
mod models;

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
#[command(propagate_version = true)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    Record(RecordArgs),
    GenCsv(GenCsvArgs),
    Test(TestArgs),
}

#[derive(clap::Args)]
struct RecordArgs {
    #[arg(long, short = 'o')]
    output_file: String,
}

#[derive(clap::Args)]
struct GenCsvArgs {
    #[arg(long, short = 'i')]
    input_wav: String,
    #[arg(long, short = 'o')]
    output_csv: String,
}

#[derive(clap::Args)]
struct TestArgs {
    #[arg(long, short = 'm')]
    model_file: String,
    #[arg(long, short = 'i')]
    input_wav: String,
    #[arg(long, short = 'd')]
    drone: bool,
}

fn record_audio(RecordArgs { output_file }: RecordArgs) {
    let running = &AtomicBool::new(true);
    thread::scope(|s| {
        let mut signals = Signals::new([SIGINT]).unwrap();
        s.spawn(move || {
            for sig in signals.forever() {
                if sig == signal_hook::consts::SIGINT {
                    running.store(false, Ordering::Relaxed);
                    println!();
                    break;
                }
            }
        });
        thread::Builder::new()
            .stack_size(1024 * 1024 * 8)
            .name("audio".to_owned())
            .spawn_scoped(s, move || {
                let audio =
                    CaptureDevice::new("hw:CARD=sndrpigooglevoi,DEV=0", 2, 48000, Format::s32());
                match audio.read(output_file, running) {
                    Ok(()) => {}
                    Err(err) => {
                        println!("Audio error: {err}");
                    }
                }
            })
            .unwrap();
    });
    println!("clean exit");
}

fn gen_csv(GenCsvArgs { input_wav, output_csv }: GenCsvArgs) {
    let mut csv = BufWriter::new(File::create(output_csv).unwrap());
    let mut reader = hound::WavReader::open(input_wav).unwrap();
    let mut samples = Vec::with_capacity(8192);

    for sample in reader.samples::<i32>() {
        let s = sample.unwrap();
        samples.push(s);
        if samples.len() == 8192 {
            let (_, values) = models::process_samples(samples.iter());

            write!(csv, "{}", values[0]).unwrap();
            for v in &values[1..] {
                write!(csv, ",{v}").unwrap();
            }
            writeln!(csv).unwrap();

            samples.clear();
        }
    }
}

fn test(TestArgs { model_file, input_wav, drone }: TestArgs) {
    let mut detection_model = models::load_onnx(model_file);

    let mut detections: CircularBuffer<20, u8> = CircularBuffer::from([0; 20]);

    let mut reader = hound::WavReader::open(input_wav).unwrap();

    let mut samples = Vec::with_capacity(8192);

    let mut predictions = 0;
    let mut correct = 0;

    for sample in reader.samples::<i32>() {
        let s = sample.unwrap();
        samples.push(s);
        if samples.len() == 8192 {
            let (_, values) = models::process_samples(samples.iter());
            samples.clear();
            let x = Array2::from_shape_vec((1, values.len()), values).unwrap();
            let mut run =
                detection_model.run(inputs![TensorRef::from_array_view(x.view()).unwrap()]);
            let Ok(ref mut outputs) = run else {
                continue;
            };
            let prob = outputs.remove("output_probability").unwrap();
            let pred = outputs["output_label"]
                .try_extract_tensor::<i64>()
                .unwrap()
                .1;
            detections.push_back(pred[0] as u8);
            drop(run);
            let pred = detections.back().unwrap();
            let prob: Sequence<DynMapValueType> = prob.into_dyn().downcast().unwrap();
            let prob = prob.extract_sequence(detection_model.allocator());
            let prob = prob
                .iter()
                .map(|p| p.try_extract_map::<i64, f32>().unwrap())
                .collect::<Vec<HashMap<i64, f32>>>();
            let prob = &prob[0].get(&(*pred as i64)).unwrap();

            let drone_predicted = detections.iter().sum::<u8>() > 1;
            predictions += 1;
            if drone_predicted == drone {
                correct += 1;
            }
            println!(
                "Drone predicted: {drone_predicted} | Drone detected: {pred} | confidence = {prob:?}"
            );
        }
    }

    println!("Acc: {}", correct as f32 / predictions as f32);
}

fn main() {
    let cli = Cli::parse();

    match cli.command {
        Commands::Record(args) => {
            record_audio(args);
        }
        Commands::GenCsv(args) => {
            gen_csv(args);
        }
        Commands::Test(args) => {
            test(args);
        }
    }

    // let mut detection_model = models::load_onnx("detection.onnx");
    //
    // let mut detections: CircularBuffer<20, u8> = CircularBuffer::from([0; 20]);
    //
    // let mut reader = hound::WavReader::open("audio.wav").unwrap();
    //
    // let mut samples = Vec::with_capacity(8192);
    //
    // for sample in reader.samples::<i32>() {
    //     let s = sample.unwrap();
    //     samples.push(s);
    //     if samples.len() == 8192 {
    //         let (_, values) = models::process_samples(samples.iter());
    //         let x = Array2::from_shape_vec((1, values.len()), values).unwrap();
    //         let mut run =
    //             detection_model.run(inputs![TensorRef::from_array_view(x.view()).unwrap()]);
    //         let Ok(ref mut outputs) = run else {
    //             samples.clear();
    //             continue;
    //         };
    //         let prob = outputs.remove("output_probability").unwrap();
    //         let pred = outputs["output_label"]
    //             .try_extract_tensor::<i64>()
    //             .unwrap()
    //             .1;
    //         detections.push_back(pred[0] as u8);
    //         drop(run);
    //         let pred = detections.back().unwrap();
    //         let prob: Sequence<DynMapValueType> = prob.into_dyn().downcast().unwrap();
    //         let prob = prob.extract_sequence(detection_model.allocator());
    //         let prob = prob
    //             .iter()
    //             .map(|p| p.try_extract_map::<i64, f32>().unwrap())
    //             .collect::<Vec<HashMap<i64, f32>>>();
    //         let prob = &prob[0].get(&(*pred as i64)).unwrap();
    //
    //         let drone_predicted = detections.iter().sum::<u8>() > 1;
    //         println!(
    //             "Drone predicted: {drone_predicted} | Drone detected: {pred} | confidence = {prob:?}"
    //         );
    //         samples.clear();
    //     }
    // }
}
