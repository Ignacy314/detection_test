use std::path::Path;

use ndarray::{Array, Array1};
use ndarray_conv::ConvExt;
use ort::session::Session;
use spectrum_analyzer::{samples_fft_to_spectrum, windows::hann_window};

pub fn process_samples<'a, I: Iterator<Item = &'a i32>>(samples: I) -> (Vec<f32>, Vec<f32>) {
    let samples = samples.map(|s| *s as f32).collect::<Vec<_>>();
    let hann_window = hann_window(&samples);

    let spectrum = samples_fft_to_spectrum(
        &hann_window,
        48000,
        spectrum_analyzer::FrequencyLimit::Range(5.0, 4000.0),
        // spectrum_analyzer::FrequencyLimit::All,
        None,
    )
    .unwrap();

    // let frequencies: Vec<f32> = (5..=4000).map(|s| s as f32).collect();

    let (freqs, values): (Vec<_>, Vec<_>) = spectrum.data().iter().copied().unzip();
    let freqs: Vec<f32> = freqs.into_iter().map(|f| f.val()).collect();

    let values: Vec<f32> = values.iter().map(|s| s.val().abs()).collect();
    let input = Array1::from_shape_vec(values.len(), values.clone()).unwrap();
    let kernel: Array1<f32> = Array::from_shape_vec(21, vec![1.0 / 21.0; 21]).unwrap();
    let output = input
        .conv(&kernel, ndarray_conv::ConvMode::Same, ndarray_conv::PaddingMode::Zeros)
        .unwrap();
    // let conv_layer = ConvolutionLayer::new(kernel, None, 1, convolutions_rs::Padding::Same);
    // let output_layer: Array3<f32> = conv_layer.convolve(&input);
    // let output_layer = output_layer.into_raw_vec();

    let mut fft_diff = values
        .iter()
        .zip(output.iter())
        .map(|(v, a)| v - a)
        .collect::<Vec<f32>>();
    let min_diff = *fft_diff.iter().min_by(|a, b| a.total_cmp(b)).unwrap();
    let max_diff = *fft_diff.iter().max_by(|a, b| a.total_cmp(b)).unwrap();

    if max_diff > min_diff {
        fft_diff
            .iter_mut()
            .for_each(|s| *s = 2.0 * (*s - min_diff) / (max_diff - min_diff) - 1.0);
    } else {
        fft_diff = vec![0.0; fft_diff.len()];
    }

    // let interp_fft_diff = interp_slice(&freqs, &fft_diff, &frequencies, &InterpMode::default());
    // assert_eq!(interp_fft_diff.len(), 3996);

    (freqs, fft_diff)
}

pub fn load_onnx<P: AsRef<Path>>(model_path: P) -> Session {
    Session::builder()
        .unwrap()
        .commit_from_file(model_path)
        .unwrap()
}

// pub fn load_detection_model<P: AsRef<Path>>(
//     model_path: P,
// ) -> RandomForestClassifier<f32, i32, Array2<f32>, Vec<i32>> {
//     bincode::deserialize_from(BufReader::new(
//         File::open(model_path).expect("Failed to open detection model path"),
//     ))
//     .expect("Failed to deserialize detection model")
// }
//
// pub fn load_location_model<P: AsRef<Path>>(
//     model_path: P,
// ) -> RandomForestRegressor<f32, f32, Array2<f32>, Vec<f32>> {
//     bincode::deserialize_from(BufReader::new(
//         File::open(model_path).expect("Failed to open location model path"),
//     ))
//     .expect("Failed to deserialize location model")
// }
