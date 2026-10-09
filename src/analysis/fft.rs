//! Spectrum analysis by FFT.

use std::{iter::Sum, ops::{AddAssign, DivAssign, MulAssign, RemAssign, SubAssign}};

use num_traits::{Float, ToPrimitive};
use rustfft::{num_complex::Complex, FftNum, FftPlanner};

use serde::Serialize;

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


/// Real-valued spectrum `value` at angular frequency `omega` \[rad/s\], as returned by
/// [`amplitude_spectrum`] and [`power_spectral_density`].
#[derive(Copy, Clone, Debug, Serialize)]
pub struct Spectrum<T> {
    /// Angular frequency \[rad/s\].
    pub omega: T,
    /// Amplitude or power spectral density, depending on the function that computed it.
    pub value: T,
}

/// Single-sided amplitude spectrum of `data` sampled with period `ts`, from one `N`-point FFT of the
/// whole record with a rectangular window (no mean removal).
///
/// Returns the bins `k = 0, ..., floor(N / 2)` (both ends included) at `omega = 2π k / (N ts)`
/// \[rad/s\], with `value = |X_k| / N` at the DC bin (`k = 0`) and at the Nyquist bin (`k = N / 2`,
/// only when `N` is even), and `2 |X_k| / N` at the other bins. A sinusoid of amplitude `A` whose
/// frequency falls exactly on a bin thus reads `A`, and a constant signal reads its value at DC.
/// Empty if `data` is empty.
pub fn amplitude_spectrum<T: FftNum + Float>(data: &[T], ts: T) -> Vec<Spectrum<T>> {
    let n = data.len();
    if n == 0 {
        return vec![];
    }
    let len = T::from(n).unwrap();
    let domega = T::from(2.0 * PI).unwrap() / (ts * len);

    let mut planner: FftPlanner<T> = FftPlanner::new();
    let fft = planner.plan_fft_forward(n);
    let mut buffer: Vec<Complex<T>> = data.iter().map(|&x| Complex::from(x)).collect();
    fft.process(&mut buffer);

    let two = T::from(2.0).unwrap();
    (0..=n / 2)
        .map(|k| {
            let amplitude = buffer[k].norm() / len;
            let value = if is_one_sided_edge(k, n) { amplitude } else { two * amplitude };
            Spectrum { omega: T::from(k).unwrap() * domega, value }
        })
        .collect()
}

/// Single-sided power spectral density of `data` sampled with period `ts` by Welch's method, with the
/// same segments as [`welch`]: Hann-windowed segments of `L` samples, about `len / n_segments` rounded
/// up to a power of two (but at least 2 and at most `len`), with 50% overlap, mean removed per segment.
///
/// Returns the bins `k = 0, ..., floor(L / 2)` at `omega = 2π k / (L ts)` \[rad/s\]. The density is
/// per **hertz**, not per rad/s, even though the frequency axis is in rad/s: its unit is (unit of
/// `data`)²/Hz. It is `|X_k|^2 / (fs Σ w_i^2)` averaged over the segments, with `fs = 1 / ts`, `X_k`
/// the FFT of a windowed segment and `w_i` the window, and doubled except at the DC bin (`k = 0`) and
/// the Nyquist bin (`k = L / 2`, only when `L` is even): the definition of
/// `scipy.signal.welch(scaling="density")`. The sum of `value * fs / L` over the bins (the density
/// times the bin width in Hz) thus approximates the variance of the signal. Empty if `data` has fewer than 2 samples.
pub fn power_spectral_density<T: FftNum + Float + Sum + AddAssign + SubAssign + DivAssign + MulAssign + RemAssign + ToPrimitive>(
    data: &[T],
    ts: T,
    n_segments: usize,
) -> Vec<Spectrum<T>> {
    if data.len() < 2 {
        return vec![];
    }
    let mut psd: Vec<T> = vec![];
    let segments = welch_segments([data], ts, n_segments, |[x], scale| {
        psd.resize(x.len() / 2 + 1, T::zero());
        for (p, xk) in psd.iter_mut().zip(x) {
            *p += xk.norm_sqr() * scale;
        }
    });

    let averager = T::one() / T::from(segments.count).unwrap();
    let two = T::from(2.0).unwrap();
    psd.into_iter()
        .enumerate()
        .map(|(k, p)| {
            let p = p * averager;
            let value = if is_one_sided_edge(k, segments.len) { p } else { two * p };
            Spectrum { omega: segments.omega(k), value }
        })
        .collect()
}

/// Whether bin `k` of an `n`-point FFT is the DC bin or the Nyquist bin, which are not doubled in a
/// single-sided spectrum.
fn is_one_sided_edge(k: usize, n: usize) -> bool {
    k == 0 || (n.is_multiple_of(2) && k == n / 2)
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
/// The power spectral density of a single signal with the same segments is
/// [`power_spectral_density`].
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
    if u.len() < 2 {
        return (vec![], vec![]);
    }

    let mut puu: Vec<T> = vec![];
    let mut pyy: Vec<T> = vec![];
    let mut pyu: Vec<Complex<T>> = vec![];
    let segments = welch_segments([u, y], ts, n_segments, |[bu, by], scale| {
        let bins = bu.len() / 2 + 1;
        puu.resize(bins, T::zero());
        pyy.resize(bins, T::zero());
        pyu.resize(bins, Complex::from(T::zero()));
        for k in 0..bins {
            let uu = bu[k].norm_sqr() * scale;
            let yy = by[k].norm_sqr() * scale;
            let yu = by[k] * bu[k].conj() * scale;

            puu[k] += uu;
            pyy[k] += yy;
            pyu[k] += yu;
        }
    });

    let bins = segments.len / 2 + 1;
    let averager = T::one() / T::from(segments.count).unwrap();
    for k in 0..bins {
        puu[k] *= averager;
        pyy[k] *= averager;
        pyu[k] *= averager;
    }

    // G = Pyu / Puu
    let mut ret: Vec<FrequencyResponse<T>> = Vec::with_capacity(bins);
    let mut coherence = vec![T::zero(); bins];

    let (floor_u, floor_y) = (power_floor(&puu), power_floor(&pyy));

    for k in 0..bins {
        let omega = segments.omega(k);
        let value = if puu[k] > floor_u { pyu[k] / puu[k] } else { Complex::from(T::zero()) };
        ret.push(FrequencyResponse { omega, value });
        coherence[k] = coherence_of(pyu[k], puu[k], pyy[k], floor_u, floor_y);
    }

    (ret, coherence)
}

/// Segmentation used by Welch's method ([`welch`], [`power_spectral_density`]).
struct WelchSegments<T> {
    /// Samples per segment `L`.
    len: usize,
    /// Number of segments averaged.
    count: usize,
    /// Sampling frequency `fs = 1 / ts`.
    fs: T,
}

impl<T: Float> WelchSegments<T> {
    /// Angular frequency `2π k fs / L` \[rad/s\] of bin `k`.
    fn omega(&self, k: usize) -> T {
        T::from(2.0 * PI).unwrap() * T::from(k).unwrap() * self.fs / T::from(self.len).unwrap()
    }
}

/// Samples per segment of Welch's method for `n` samples split into about `n_segments` segments:
/// `2^ceil(log2(n / n_segments))`, at least 2 (for the window) and at most `n`.
fn welch_segment_len(n: usize, n_segments: usize) -> usize {
    n.div_ceil(n_segments.max(1)).next_power_of_two().clamp(2, n)
}

/// Symmetric Hann window `w_i = (1 - cos(2π i / (len - 1))) / 2` of `len` (at least 2) samples.
fn hann_window<T: Float>(len: usize) -> Vec<T> {
    (0..len).map(|i| {
        T::from(0.5).unwrap() * (T::one() - (T::from(2.0 * PI).unwrap() * T::from(i).unwrap() / T::from(len - 1).unwrap()).cos())
    }).collect()
}

/// Splits each of the `signals` (of a common length, at least 2) into the segments of Welch's method,
/// removes the mean of each segment, applies the Hann window and takes the FFT. For each segment,
/// `accumulate` gets the FFTs of the signals (all `L` bins) and the density scale `1 / (fs Σ w_i^2)`.
fn welch_segments<T, const M: usize>(
    signals: [&[T]; M],
    ts: T,
    n_segments: usize,
    mut accumulate: impl FnMut(&[Vec<Complex<T>>; M], T),
) -> WelchSegments<T>
where
    T: FftNum + Float + Sum + AddAssign + DivAssign,
{
    let n = signals[0].len();
    let fs = T::one() / ts;
    let nperseg = welch_segment_len(n, n_segments);
    let noverlap = nperseg / 2;

    let window = hann_window::<T>(nperseg);
    let scale = T::one() / (fs * window.iter().map(|&w| w*w).sum::<T>());

    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(nperseg);

    let step = nperseg - noverlap;
    let mut num_segments = 0;

    let mut buffers: [Vec<Complex<T>>; M] = std::array::from_fn(|_| vec![Complex::from(T::zero()); nperseg]);

    for start in (0..=n - nperseg).step_by(step) {
        for (signal, buffer) in signals.iter().zip(buffers.iter_mut()) {
            let segment = &signal[start..start + nperseg];
            let mut mean = T::zero();
            for &x in segment {
                mean += x;
            }
            mean /= T::from(nperseg).unwrap();

            for ((b, &x), &w) in buffer.iter_mut().zip(segment).zip(&window) {
                *b = Complex::from((x - mean) * w);
            }

            // FFT
            fft.process(buffer);
        }
        accumulate(&buffers, scale);
        num_segments += 1;
    }

    WelchSegments { len: nperseg, count: num_segments, fs }
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
