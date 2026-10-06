//! Frequency response of the model against the nonparametric estimate from the data.

use num_complex::Complex;
use num_traits::Float;
use rustfft::FftNum;

use super::{Validation, ValidationError};

impl<T: Float + FftNum> Validation<T> {
    /// Frequency response of the model compared with the nonparametric estimate from the data,
    /// for any input, by the same segments as `coherence_test`:
    ///
    /// ```text
    /// measured  G(f) = S_yu(f) / S_uu(f)              (H1 estimate)
    /// model     Ĝ(f) = S_ŷu(f) / S_uu(f) = G(f) - S_εu(f) / S_uu(f)
    /// ```
    ///
    /// Both are estimated with the same windowed segments, so the leakage of the window affects
    /// them alike and their difference is exactly `S_εu / S_uu` (the model response includes the
    /// input delay, from the simulation). Both are therefore windowed estimates of the response of
    /// the sampled system (with the hold), not `Ĝ(jω)`: their bias, as for any H1 estimate, falls
    /// as the segments grow against the impulse response of the plant. The standard deviation of the measured response is
    /// `σ(f) = sqrt((1 - γ²_yu) S_yy / (L S_uu))` (`γ²_yu` the coherence of the output and the
    /// input, `L` the equivalent number of independent segments): `G ± 2σ` is a ~95 % band, and
    /// `|Ĝ - G|^2 / σ^2` is about `χ²(2) / 2` (mean 1) for a right model. Where the leakage is
    /// strong (sharp resonances against short segments) it also lowers `γ²_yu` and so inflates
    /// `σ`: the normalized error is then conservative (below 1 for a right model).
    ///
    /// This is a diagnostic rather than another test (it carries the same information as
    /// `coherence_test`): it shows the model error as a gain \[dB\] and a phase error against the
    /// uncertainty of the data, e.g. for a Bode plot of both.
    pub fn frequency_response(&self, segment_len: usize) -> Result<FrequencyResponseComparison<T>, ValidationError> {
        let spectra = self.spectra(segment_len)?;
        let l = spectra.effective_segments;
        let zero = Complex::new(T::zero(), T::zero());
        let bins: Vec<usize> = (1..=segment_len / 2).collect();
        let ratio = |a: Complex<T>, f: usize| if spectra.uu[f] > T::zero() { a / spectra.uu[f] } else { zero };
        Ok(FrequencyResponseComparison {
            measured: bins.iter().map(|&f| ratio(spectra.yu[f], f)).collect(),
            model: bins.iter().map(|&f| ratio(spectra.yu[f] - spectra.eu[f], f)).collect(),
            coherence: bins
                .iter()
                .map(|&f| {
                    let denom = spectra.uu[f] * spectra.yy[f];
                    if denom > T::zero() { spectra.yu[f].norm_sqr() / denom } else { T::zero() }
                })
                .collect(),
            stdev: bins
                .iter()
                .map(|&f| {
                    if spectra.uu[f] > T::zero() {
                        let noise = spectra.yy[f] - spectra.yu[f].norm_sqr() / spectra.uu[f]; // (1 - γ²) S_yy
                        (noise.max(T::zero()) / (l * spectra.uu[f])).sqrt()
                    } else {
                        T::infinity()
                    }
                })
                .collect(),
            input_power: bins.iter().map(|&f| spectra.uu[f]).collect(),
            bins,
            segment_len,
            effective_segments: l,
        })
    }
}

/// Result of `Validation::frequency_response`.
#[derive(Clone, Debug)]
pub struct FrequencyResponseComparison<T> {
    /// Frequency bins `f` (frequency `f / (segment_len ts)`).
    pub bins: Vec<usize>,
    /// Nonparametric estimate `G(f) = S_yu / S_uu` from the data.
    pub measured: Vec<Complex<T>>,
    /// Response of the model `Ĝ(f) = S_ŷu / S_uu`, estimated with the same segments.
    pub model: Vec<Complex<T>>,
    /// Coherence `γ²_yu(f)` of the output and the input (1 without noise and nonlinearity).
    pub coherence: Vec<T>,
    /// Standard deviation `σ(f)` of the measured response (complex, `E|G - G0|^2 = σ^2`).
    pub stdev: Vec<T>,
    /// Input power at each bin, for `excited`.
    pub input_power: Vec<T>,
    /// Segment length \[samples\].
    pub segment_len: usize,
    /// Equivalent number of independent segments `L`.
    pub effective_segments: T,
}

impl<T: Float> FrequencyResponseComparison<T> {
    /// Frequencies \[Hz\] of the bins for the sampling period `ts`.
    pub fn frequencies(&self, ts: T) -> Vec<T> {
        let df = T::one() / (T::from(self.segment_len).unwrap() * ts);
        self.bins.iter().map(|&f| T::from(f).unwrap() * df).collect()
    }

    /// The bins where the input power is at least `relative` times its maximum: elsewhere the
    /// measured response is not defined by the data.
    pub fn excited(&self, relative: T) -> Self {
        let threshold = relative * self.input_power.iter().fold(T::zero(), |acc, &p| acc.max(p));
        let keep: Vec<usize> = (0..self.bins.len()).filter(|&i| self.input_power[i] >= threshold).collect();
        Self {
            bins: keep.iter().map(|&i| self.bins[i]).collect(),
            measured: keep.iter().map(|&i| self.measured[i]).collect(),
            model: keep.iter().map(|&i| self.model[i]).collect(),
            coherence: keep.iter().map(|&i| self.coherence[i]).collect(),
            stdev: keep.iter().map(|&i| self.stdev[i]).collect(),
            input_power: keep.iter().map(|&i| self.input_power[i]).collect(),
            ..self.clone()
        }
    }

    /// Relative error `|Ĝ - G| / |G|` at each bin.
    pub fn relative_error(&self) -> Vec<T> {
        self.model.iter().zip(&self.measured).map(|(m, g)| (m - g).norm() / g.norm()).collect()
    }

    /// RMS of the relative error over the bins.
    pub fn rms_relative_error(&self) -> T {
        let e = self.relative_error();
        (e.iter().fold(T::zero(), |acc, &v| acc + v * v) / T::from(e.len().max(1)).unwrap()).sqrt()
    }

    /// Gain error `20 log10 |Ĝ / G|` \[dB\] at each bin.
    pub fn gain_error_db(&self) -> Vec<T> {
        let twenty = T::from(20).unwrap();
        self.model.iter().zip(&self.measured).map(|(m, g)| twenty * (m.norm() / g.norm()).log10()).collect()
    }

    /// Phase error `arg(Ĝ / G)` \[rad\], in `(-π, π]`, at each bin.
    pub fn phase_error(&self) -> Vec<T> {
        self.model.iter().zip(&self.measured).map(|(m, g)| (m / g).arg()).collect()
    }

    /// Model error in units of the uncertainty of the data, `|Ĝ - G|^2 / σ^2` at each bin: about
    /// `χ²(2) / 2` (mean 1) for a right model; much larger means a model error beyond the noise.
    pub fn normalized_error(&self) -> Vec<T> {
        self.model.iter().zip(&self.measured).zip(&self.stdev).map(|((m, g), s)| (m - g).norm_sqr() / (*s * *s)).collect()
    }

    /// Mean of `normalized_error` over the bins (about 1 for a right model).
    pub fn mean_normalized_error(&self) -> T {
        let e = self.normalized_error();
        e.iter().fold(T::zero(), |acc, &v| acc + v) / T::from(e.len().max(1)).unwrap()
    }
}
