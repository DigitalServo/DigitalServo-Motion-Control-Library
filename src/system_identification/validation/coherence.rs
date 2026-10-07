//! Coherence test of the residual and the input, for any input.

use num_traits::Float;
use rustfft::FftNum;

use super::{Validation, ValidationError};
use crate::analysis::fft::{coherence_of, power_floor};

impl<T: Float + FftNum> Validation<T> {
    /// Coherence test of the residual and the input, for any input (non-periodic: random, chirp,
    /// measured operation data, ...).
    ///
    /// The evaluated samples are split into segments of `segment_s` \[s\] (`segment_len =
    /// segment_s / ts` samples, a whole number; Hann window, 50 % overlap, mean removed per
    /// segment), and with the DFTs `E_l(f)`, `U_l(f)` of segment `l`
    ///
    /// ```text
    /// γ²(f) = |Σ_l E_l(f) U_l(f)*|^2 / (Σ_l |E_l(f)|^2  Σ_l |U_l(f)|^2)
    /// ```
    ///
    /// at the bins `f = 1 ..= segment_len / 2` (frequency `f / segment_s` \[Hz\]) where the input
    /// has power (see below). If the model is
    /// right, the residual is noise independent of the input, and for `L` independent segments
    /// `γ²(f)` follows the Beta distribution `Beta(1, L - 1)` whatever the input (it is the squared
    /// projection of the Gaussian vector `E_l(f)` on the fixed direction `U_l(f)`): the bound at
    /// the given `confidence` is `1 - (1 - confidence)^(1 / (L - 1))`, and the mean `1 / L`.
    /// Overlapping segments are not independent; `L` is the equivalent number of independent
    /// segments for the window and the overlap (Welch: `K / (1 + 2 Σ_m (1 - m/K) c_m^2)` for `K`
    /// segments, `c_m` the normalized overlap of the window with itself shifted by `m` steps).
    ///
    /// `γ²(f)` outside the bound means the residual still depends on the input at that frequency.
    /// Longer segments give a finer frequency resolution but fewer segments, so a higher bound
    /// (less power); at the frequencies where the input has no power the test cannot detect
    /// anything (see `CoherenceTest::excited`). The bins where the input power is at most `1e-12`
    /// times its maximum over the bins (numerical noise) are left out, and `γ²(f)` is zero where
    /// the power of the residual is at most `1e-12` times its maximum (e.g. a residual that is
    /// identically zero).
    pub fn coherence_test(&self, segment_s: T, confidence: T) -> Result<CoherenceTest<T>, ValidationError> {
        let segment_len = self.duration_samples(segment_s)?;
        let spectra = self.spectra(segment_len)?;
        let l = spectra.effective_segments;
        let (floor_e, floor_u) = (power_floor(&spectra.ee), power_floor(&spectra.uu));
        let bins: Vec<usize> = (1..=segment_len / 2).filter(|&f| spectra.uu[f] > floor_u).collect();
        Ok(CoherenceTest {
            coherence: bins.iter().map(|&f| coherence_of(spectra.eu[f], spectra.ee[f], spectra.uu[f], floor_e, floor_u)).collect(),
            input_power: bins.iter().map(|&f| spectra.uu[f]).collect(),
            bins,
            segment_len,
            ts: self.ts,
            segments: spectra.segments,
            effective_segments: l,
            bound: T::one() - (T::one() - confidence).powf(T::one() / (l - T::one())),
        })
    }
}

/// Result of `Validation::coherence_test`.
#[derive(Clone, Debug)]
pub struct CoherenceTest<T> {
    /// Frequency bins `f` (frequency `f / (segment_len ts)` \[Hz\]) where the input has power.
    pub bins: Vec<usize>,
    /// Coherence `γ²(f)` of the residual and the input at each bin.
    pub coherence: Vec<T>,
    /// Input power at each bin (mean `|U_l(f)|^2` over the segments, windowed), for `excited`.
    pub input_power: Vec<T>,
    /// Segment length \[samples\].
    pub segment_len: usize,
    /// Sampling period \[s\].
    pub ts: T,
    /// Number of (overlapping) segments `K`.
    pub segments: usize,
    /// Equivalent number of independent segments `L`.
    pub effective_segments: T,
    /// Bound of `γ²(f)` at the given confidence.
    pub bound: T,
}

impl<T: Float> CoherenceTest<T> {
    /// Frequencies \[Hz\] of the bins.
    pub fn frequencies(&self) -> Vec<T> {
        let df = T::one() / (T::from(self.segment_len).unwrap() * self.ts);
        self.bins.iter().map(|&f| T::from(f).unwrap() * df).collect()
    }

    /// The bins where the input power is at least `relative` times its maximum (e.g. `1e-2` for
    /// -20 dB): elsewhere the test has no power to detect a model error.
    pub fn excited(&self, relative: T) -> Self {
        let threshold = relative * self.input_power.iter().fold(T::zero(), |acc, &p| acc.max(p));
        let keep: Vec<usize> = (0..self.bins.len()).filter(|&i| self.input_power[i] >= threshold).collect();
        Self {
            bins: keep.iter().map(|&i| self.bins[i]).collect(),
            coherence: keep.iter().map(|&i| self.coherence[i]).collect(),
            input_power: keep.iter().map(|&i| self.input_power[i]).collect(),
            ..self.clone()
        }
    }

    /// Bins with `γ²(f) > bound`.
    pub fn outside(&self) -> Vec<usize> {
        self.bins.iter().zip(&self.coherence).filter(|(_, c)| **c > self.bound).map(|(&f, _)| f).collect()
    }

    /// Fraction of the bins with `γ²(f) > bound`; about `1 - confidence` for a right model.
    pub fn fraction_outside(&self) -> T {
        T::from(self.outside().len()).unwrap() / T::from(self.bins.len().max(1)).unwrap()
    }

    /// Mean of `γ²(f)` over the bins, to compare with `expected_coherence`.
    pub fn mean_coherence(&self) -> T {
        self.coherence.iter().fold(T::zero(), |acc, &c| acc + c) / T::from(self.bins.len().max(1)).unwrap()
    }

    /// Expectation `1 / L` of `γ²(f)` for a right model.
    pub fn expected_coherence(&self) -> T {
        T::one() / self.effective_segments
    }
}
