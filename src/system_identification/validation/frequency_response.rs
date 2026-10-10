//! Frequency response of the model against the nonparametric estimate from the data.

use num_complex::Complex;
use num_traits::Float;
use rustfft::FftNum;

use super::{in_band, Validation, ValidationError};
use crate::analysis::fft::{coherence_of, power_floor};

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
    /// uncertainty of the data, e.g. for a Bode plot of both. The bins where the input power is at
    /// most `1e-12` times its maximum over the bins (numerical noise) are left out: the data do not
    /// define the response there.
    pub fn frequency_response(&self, segment_s: T) -> Result<FrequencyResponseComparison<T>, ValidationError> {
        let segment_len = self.duration_samples(segment_s)?;
        let spectra = self.spectra(segment_len)?;
        let l = spectra.effective_segments;
        let (floor_u, floor_y) = (power_floor(&spectra.uu), power_floor(&spectra.yy));
        let bins: Vec<usize> = (1..=segment_len / 2).filter(|&f| spectra.uu[f] > floor_u).collect();
        let ratio = |a: Complex<T>, f: usize| a / spectra.uu[f];
        Ok(FrequencyResponseComparison {
            measured: bins.iter().map(|&f| ratio(spectra.yu[f], f)).collect(),
            model: bins.iter().map(|&f| ratio(spectra.yu[f] - spectra.eu[f], f)).collect(),
            coherence: bins.iter().map(|&f| coherence_of(spectra.yu[f], spectra.uu[f], spectra.yy[f], floor_u, floor_y)).collect(),
            stdev: bins
                .iter()
                .map(|&f| {
                    let coherent = spectra.yu[f].norm() / spectra.uu[f].sqrt();
                    let noise = spectra.yy[f] - coherent * coherent; // (1 - γ²) S_yy
                    (noise.max(T::zero()) / (l * spectra.uu[f])).sqrt()
                })
                .collect(),
            input_power: bins.iter().map(|&f| spectra.uu[f]).collect(),
            bins,
            segment_len,
            ts: self.ts,
            effective_segments: l,
        })
    }
}

/// Result of `Validation::frequency_response`.
#[derive(Clone, Debug)]
pub struct FrequencyResponseComparison<T> {
    /// Frequency bins `f` (frequency `f / (segment_len ts)` \[Hz\]) where the input has power.
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
    /// Sampling period \[s\].
    pub ts: T,
    /// Equivalent number of independent segments `L`.
    pub effective_segments: T,
}

impl<T: Float> FrequencyResponseComparison<T> {
    /// Frequencies \[Hz\] of the bins.
    pub fn frequencies(&self) -> Vec<T> {
        let df = T::one() / (T::from(self.segment_len).unwrap() * self.ts);
        self.bins.iter().map(|&f| T::from(f).unwrap() * df).collect()
    }

    /// The bins where the input power is at least `relative` times its maximum: elsewhere the
    /// measured response is not defined by the data.
    pub fn excited(&self, relative: T) -> Self {
        let threshold = relative * self.input_power.iter().fold(T::zero(), |acc, &p| acc.max(p));
        self.select(|i| self.input_power[i] >= threshold)
    }

    /// The bins with frequencies in `[low, high]` \[Hz\] (both included), as
    /// `CoherenceTest::band`.
    pub fn band(&self, (low, high): (T, T)) -> Self {
        let in_band = in_band(self.segment_len, self.ts, low, high);
        self.select(|i| in_band(self.bins[i]))
    }

    fn select(&self, keep: impl Fn(usize) -> bool) -> Self {
        let keep: Vec<usize> = (0..self.bins.len()).filter(|&i| keep(i)).collect();
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
