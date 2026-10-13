//! Spectrogram of a linear chirp from 10 Hz to 400 Hz over 2 s, sampled at 1 kHz with 0.1 s windows
//! (`out/spectrogram.csv`, one row per time and frequency).
//!
//! `cargo run --example spectrogram`

use std::f64::consts::PI;

use dsmc::fft::spectrogram;
use dsmc::logger::DataStorage;
use serde::Serialize;

#[derive(Serialize)]
struct Row {
    time: f64,
    freq_hz: f64,
    psd: f64,
}

fn main() {
    let (ts, duration, window_time) = (1e-3, 2.0, 0.1);
    let (f0, f1) = (10.0, 400.0);
    let rate = (f1 - f0) / duration;
    let n = (duration / ts) as usize;
    let data: Vec<f64> = (0..n)
        .map(|k| {
            let t = k as f64 * ts;
            (2.0 * PI * (f0 * t + 0.5 * rate * t * t)).sin()
        })
        .collect();

    let s = spectrogram(&data, ts, window_time);

    let mut storage = DataStorage::new("./out/spectrogram.csv").unwrap();
    for (&time, row) in s.time.iter().zip(&s.psd) {
        for (&omega, &psd) in s.omega.iter().zip(row) {
            storage.add(&Row { time, freq_hz: omega / (2.0 * PI), psd }).unwrap();
        }
    }
    storage.close().unwrap();
}
