//! Segment spectra of the residual, the input and the output (Welch), shared by the coherence test
//! and the frequency-response comparison.

use num_complex::Complex;
use num_traits::Float;
use rustfft::{FftNum, FftPlanner};

use super::{Validation, ValidationError};

impl<T: Float + FftNum> Validation<T> {
    /// Mean auto / cross spectra over the segments of `segment_len` samples (Hann window, 50 %
    /// overlap, mean removed per segment) of the residual `ε`, the input `u` and the output `y`,
    /// at the bins `0 ..= segment_len / 2`, and the equivalent number of independent segments.
    pub(super) fn spectra(&self, segment_len: usize) -> Result<Spectra<T>, ValidationError> {
        let start = self.start.min(self.residual.len());
        let (e, u) = (&self.residual[start..], &self.input[start..]);
        let y: Vec<T> = self.simulated[start..].iter().zip(e).map(|(&a, &b)| a + b).collect();
        let step = (segment_len / 2).max(1);
        let segments = if segment_len >= 2 && e.len() >= segment_len { (e.len() - segment_len) / step + 1 } else { 0 };
        if segments < 2 {
            return Err(ValidationError::TooFewSegments { segments, segment_len });
        }

        let two_pi = T::from(2.0 * std::f64::consts::PI).unwrap();
        let window: Vec<T> = (0..segment_len)
            .map(|i| T::from(0.5).unwrap() * (T::one() - (two_pi * T::from(i).unwrap() / T::from(segment_len - 1).unwrap()).cos()))
            .collect();
        let fft = FftPlanner::new().plan_fft_forward(segment_len);
        let half = segment_len / 2;
        let zero = Complex::new(T::zero(), T::zero());
        let mut spectra = Spectra {
            ee: vec![T::zero(); half + 1],
            uu: vec![T::zero(); half + 1],
            yy: vec![T::zero(); half + 1],
            eu: vec![zero; half + 1],
            yu: vec![zero; half + 1],
            segments,
            effective_segments: T::zero(),
        };
        let transformed = |x: &[T]| {
            let mean = x.iter().fold(T::zero(), |acc, &v| acc + v) / T::from(x.len()).unwrap();
            let mut buffer: Vec<Complex<T>> = x.iter().zip(&window).map(|(&v, &w)| Complex::new((v - mean) * w, T::zero())).collect();
            fft.process(&mut buffer);
            buffer
        };
        for l in 0..segments {
            let range = l * step..l * step + segment_len;
            let (eb, ub, yb) = (transformed(&e[range.clone()]), transformed(&u[range.clone()]), transformed(&y[range]));
            for f in 0..=half {
                spectra.ee[f] = spectra.ee[f] + eb[f].norm_sqr();
                spectra.uu[f] = spectra.uu[f] + ub[f].norm_sqr();
                spectra.yy[f] = spectra.yy[f] + yb[f].norm_sqr();
                spectra.eu[f] = spectra.eu[f] + eb[f] * ub[f].conj();
                spectra.yu[f] = spectra.yu[f] + yb[f] * ub[f].conj();
            }
        }
        let k = T::from(segments).unwrap();
        for f in 0..=half {
            spectra.ee[f] = spectra.ee[f] / k;
            spectra.uu[f] = spectra.uu[f] / k;
            spectra.yy[f] = spectra.yy[f] / k;
            spectra.eu[f] = spectra.eu[f] / k;
            spectra.yu[f] = spectra.yu[f] / k;
        }

        // Equivalent number of independent segments
        let energy = window.iter().fold(T::zero(), |acc, &w| acc + w * w);
        let correlation = (1..segments).take_while(|m| m * step < segment_len).fold(T::zero(), |acc, m| {
            let c = (0..segment_len - m * step).fold(T::zero(), |acc, t| acc + window[t] * window[t + m * step]) / energy;
            acc + (T::one() - T::from(m).unwrap() / k) * c * c
        });
        spectra.effective_segments = k / (T::one() + T::from(2).unwrap() * correlation);
        Ok(spectra)
    }
}

/// Mean spectra of `Validation::spectra` (windowed DFT units).
pub(super) struct Spectra<T> {
    pub(super) ee: Vec<T>,
    pub(super) uu: Vec<T>,
    pub(super) yy: Vec<T>,
    pub(super) eu: Vec<Complex<T>>,
    pub(super) yu: Vec<Complex<T>>,
    pub(super) segments: usize,
    pub(super) effective_segments: T,
}
