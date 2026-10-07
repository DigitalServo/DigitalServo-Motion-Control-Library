//! Spectrum analysis by FFT.

use std::{iter::Sum, ops::{AddAssign, DivAssign, MulAssign, RemAssign, SubAssign}};

use num_traits::{Float, ToPrimitive};
use rustfft::{num_complex::Complex, FftNum, FftPlanner};

use crate::analysis::FrequencyResponse;

use std::f64::consts::PI;

/// Single-sided spectrum of `data` sampled with period `ts`: the first `N / 2` FFT bins, scaled by `1 / N`,
/// at `omega = 2π k / (N ts)` \[rad/s\].
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


/// Frequency response `G = P_yu / P_uu` from input `u` and output `y` (sampled with period `ts`) by
/// Welch's method: Hann-windowed segments of about `len / n_segments` samples (rounded up to a power
/// of two, but at most `len`) with 50% overlap, mean removed per segment.
///
/// Returns the response at `omega` \[rad/s\] and the coherence `|P_yu|^2 / (P_uu P_yy)` per
/// frequency. A bin counts as unexcited, with zero response and coherence, where `P_uu` is at most
/// `1e-12` times its maximum over the bins (and the coherence is also zero where `P_yy` is at most
/// `1e-12` times its maximum): the thresholds are relative, so the result does not depend on the
/// units of `u` and `y`. Both are empty if `u` has fewer than 2 samples. The response can be passed
/// to the identification methods in
/// [`system_identification::frequency_response`](crate::system_identification::frequency_response).
///
/// # Panics
///
/// If `u` and `y` have different lengths.
pub fn welch<T: FftNum + Float + Sum + AddAssign + SubAssign + DivAssign + MulAssign + RemAssign + ToPrimitive>(
    u: &[T],
    y: &[T],
    ts: T,
    n_segments: usize,
) -> (Vec<FrequencyResponse<T>>, Vec<T>) {
    assert_eq!(u.len(), y.len(), "welch: input and output lengths differ");
    let n = u.len();
    if n < 2 {
        return (vec![], vec![]);
    }
    let fs = T::one() / ts;
    // 2^ceil(log2(n / n_segments)), at least 2 (for the window) and at most n
    let nperseg = n.div_ceil(n_segments.max(1)).next_power_of_two().clamp(2, n);
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

    let (floor_u, floor_y) = (power_floor(&puu), power_floor(&pyy));

    for k in 0..=nperseg / 2 {
        let omega = T::from(2.0 * PI).unwrap() * T::from(k).unwrap() * fs / T::from(nperseg).unwrap();
        let value = if puu[k] > floor_u { pyu[k] / puu[k] } else { Complex::from(T::zero()) };
        ret.push(FrequencyResponse { omega, value });
        coherence[k] = coherence_of(pyu[k], puu[k], pyy[k], floor_u, floor_y);
    }

    (ret, coherence)
}

/// Power below which a bin of the spectrum `p` counts as unexcited: `1e-12` times its maximum over
/// the bins (the numerical noise of the window leakage). Relative, so that it does not depend on the
/// units of the signal; zero for a signal that is identically zero, so that `p > floor` still
/// rejects every bin.
pub(crate) fn power_floor<T: Float>(p: &[T]) -> T {
    T::from(1e-12).unwrap() * p.iter().copied().fold(T::zero(), T::max)
}

/// Coherence `|P_ab|^2 / (P_aa P_bb)`, zero where `P_aa` or `P_bb` is at most its floor. Computed as
/// `(|P_ab| / sqrt(P_aa) / sqrt(P_bb))^2` so that the products of small powers do not underflow.
pub(crate) fn coherence_of<T: Float>(ab: Complex<T>, aa: T, bb: T, floor_a: T, floor_b: T) -> T {
    if aa > floor_a && bb > floor_b {
        let c = ab.norm() / aa.sqrt() / bb.sqrt();
        c * c
    } else {
        T::zero()
    }
}
