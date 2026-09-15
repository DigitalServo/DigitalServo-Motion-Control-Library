use std::{iter::Sum, ops::{AddAssign, DivAssign, MulAssign, RemAssign, SubAssign}};

use num_traits::{Float, ToPrimitive};
use rustfft::{num_complex::Complex, FftNum, FftPlanner};

use crate::analysis::FrequencyResponse;

use std::f64::consts::PI;

pub fn fft<T: FftNum + Float>(data: &[T], ts: T) -> Vec<FrequencyResponse<T>> {

    let data_len: usize = data.len();
    let (domega, inv_data_len) = match T::from(data.len()) {
        Some(len) => {
            let dfreq = T::one() / (ts * len);
            let domega = T::from(2.0 * PI).unwrap() * dfreq;
            (domega, T::one() / len)
        },
        None => return vec![],
    };

    let mut planner: FftPlanner<T> = FftPlanner::new();
    let fft = planner.plan_fft_forward(data_len);

    let mut buffer: Vec<Complex<T>> = data
        .iter()
        .map(|&x| Complex::from(x * inv_data_len))
        .collect();

    fft.process(&mut buffer);

    let data_len_half: usize = buffer.len() / 2;
    buffer[0..data_len_half]
        .iter()
        .enumerate()
        .map(|(i, &value)| {
            let omega = T::from(i).unwrap() * domega;
            FrequencyResponse { omega, value }
        })
        .collect()
}


pub fn welch<T: FftNum + Float + Sum + AddAssign + SubAssign + DivAssign + MulAssign + RemAssign + ToPrimitive>(
    u: &[T],
    y: &[T],
    ts: T,
    n_segments: usize,
) -> (Vec<FrequencyResponse<T>>, Vec<T>) {
    let n = u.len();
    let fs = T::one() / ts;
    let nperseg = {
        let x = (T::from(n).unwrap() / T::from(n_segments).unwrap()).log2().ceil();
        2usize.pow(x.to_u32().unwrap())
    };
    let noverlap = nperseg / 2;

    let create_hann_window = |len: usize| -> Vec<T> {
        (0..len).map(|i| {
            T::from(0.5).unwrap() * (T::one() - (T::from(2.0 * PI).unwrap() * T::from(i).unwrap() / T::from(len - 1).unwrap()).cos())
        }).collect()
    };

    let window = create_hann_window(nperseg);
    let scale = T::one() / (fs * window.iter().map(|&w| w*w).sum::<T>());

    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(nperseg);

    let mut puu = vec![T::zero(); nperseg / 2 + 1];
    let mut pyy = vec![T::zero(); nperseg / 2 + 1];
    let mut pyu = vec![Complex::from(T::zero()); nperseg / 2 + 1];

    let step = nperseg - noverlap;
    let mut num_segments = 0;

    let mut buffer_u = vec![Complex::from(T::zero()); nperseg];
    let mut buffer_y = vec![Complex::from(T::zero()); nperseg];

    for start in (0..=n - nperseg).step_by(step) {
        let mut mean_u = T::zero();
        let mut mean_y = T::zero();
        for i in 0..nperseg {
            let idx = start + i;
            mean_u += u[idx];
            mean_y += y[idx];
        }
        mean_u /= T::from(nperseg).unwrap();
        mean_y /= T::from(nperseg).unwrap();

        for i in 0..nperseg {
            let idx = start + i;
            buffer_u[i] = Complex::from((u[idx] - mean_u) * window[i]);
            buffer_y[i] = Complex::from((y[idx] - mean_y) * window[i]);
        }

        // FFT
        fft.process(&mut buffer_u);
        fft.process(&mut buffer_y);

        for k in 0..=nperseg / 2 {
            let uu = buffer_u[k].norm_sqr() * scale;
            let yy = buffer_y[k].norm_sqr() * scale;
            let yu = buffer_y[k] * buffer_u[k].conj() * scale;

            puu[k] += uu;
            pyy[k] += yy;
            pyu[k] += yu;
        }
        num_segments += 1;
    }

    let averager = T::one() / T::from(num_segments).unwrap();
    for k in 0..=nperseg / 2 {
        puu[k] *= averager;
        pyy[k] *= averager;
        pyu[k] *= averager;
    }

    // G = Pyu / Puu
    let mut ret: Vec<FrequencyResponse<T>> = Vec::with_capacity(nperseg / 2 + 1);
    let mut coherence = vec![T::zero(); nperseg / 2 + 1];

    let tol = T::from(1e-12).unwrap();

    for k in 0..=nperseg / 2 {
        let omega = T::from(2.0 * PI).unwrap() * T::from(k).unwrap() * fs / T::from(nperseg).unwrap();
        let value = if puu[k] > tol { pyu[k] / puu[k] } else { Complex::from(T::zero()) };
        ret.push(FrequencyResponse { omega, value });

        let denom = puu[k] * pyy[k];
        coherence[k] = if denom > tol { pyu[k].norm_sqr() / denom } else { T::zero() };
    }

    (ret, coherence)
}
