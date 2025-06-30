use std::{fs::File, io::BufWriter, path::Path, sync::atomic::AtomicBool};

use alsa::{
    Direction, Error, ValueOr,
    pcm::{Access, Format, HwParams, PCM},
};

pub struct CaptureDevice {
    device_name: String,
    channels: u32,
    samplerate: u32,
    format: Format,
}

impl CaptureDevice {
    pub fn new(device_name: &str, channels: u32, samplerate: u32, format: Format) -> Self {
        Self {
            device_name: device_name.to_owned(),
            channels,
            samplerate,
            format,
        }
    }

    fn init_device(&self) -> Result<PCM, Error> {
        let pcm = PCM::new(&self.device_name, Direction::Capture, false)?;
        {
            let hwp = HwParams::any(&pcm)?;
            hwp.set_channels(self.channels)?;
            hwp.set_rate(self.samplerate, ValueOr::Nearest)?;
            hwp.set_format(self.format)?;
            hwp.set_access(Access::RWInterleaved)?;
            let buf_size = hwp.get_buffer_size_max()?;
            hwp.set_buffer_size(buf_size)?;
            pcm.hw_params(&hwp)?;
        }
        pcm.prepare()?;
        pcm.start()?;
        Ok(pcm)
    }

    pub fn read<P: AsRef<Path>>(&self, output_file: P, running: &AtomicBool) -> Result<(), Error> {
        let pcm = self.init_device()?;
        let io = match &self.format {
            Format::S32LE | Format::S32BE => pcm.io_i32()?,
            _ => return Err(Error::new("Format unimplemented", 0)),
        };

        let mut buf = [0i32; 1024 * 32];

        let mut writer = hound::WavWriter::new(
            BufWriter::new(File::create(output_file).unwrap()),
            hound::WavSpec {
                channels: self.channels as u16,
                sample_rate: self.samplerate,
                bits_per_sample: 32,
                sample_format: hound::SampleFormat::Int,
            },
        )
        .unwrap();

        println!("start audio read");
        while running.load(std::sync::atomic::Ordering::Relaxed) {
            match io.readi(&mut buf) {
                Ok(s) => {
                    let n = s * self.channels as usize;
                    for sample in &buf[..n] {
                        writer.write_sample(*sample).unwrap();
                    }
                }
                Err(err) => {
                    if err.errno() != 11 {
                        println!("ALSA try recover from: {err}");
                        pcm.try_recover(err, false)?;
                    }
                }
            }
            // thread::sleep(Duration::from_millis(1).saturating_sub(start.elapsed()));
        }
        Ok(())
    }
}
