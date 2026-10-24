//! System analysis tools.

use std::{marker::PhantomData, ops::Deref};

use nalgebra::Complex;

mod bode_diagram;
mod frequency_transfer_function;
mod frequency_unit;
mod nyquist;
mod statistics;
pub mod fft;

pub use bode_diagram::BodeDiagramPlotter;
pub use frequency_transfer_function::FrequencyTransferFunction;
pub use frequency_unit::{FrequencyUnit, Hz, RadPerSec};
pub use nyquist::NyquistPlotter;
pub use statistics::Statistics;

use num_traits::{Float, FloatConst};
use serde::{ser::SerializeStruct, Serialize, Serializer};

/// Complex frequency response `value = G(jω)` at angular frequency `omega` [rad/s].
#[derive(Copy, Clone)]
pub struct FrequencyResponse<T> {
    /// Angular frequency [rad/s].
    pub omega: T,
    /// `G(jω)`.
    pub value: Complex<T>
}

/// Serialized flat as `[omega, re, im]`.
impl<T: Serialize> Serialize for FrequencyResponse<T> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut state = serializer.serialize_struct("FrequencyResponse", 3)?;
        state.serialize_field("omega", &self.omega)?;
        state.serialize_field("re", &self.value.re)?;
        state.serialize_field("im", &self.value.im)?;
        state.end()
    }
}

/// Frequency response `value = G(jω)` at angular frequency `omega` [rad/s] with its coherence (from
/// [`welch`](fft::welch)).
#[derive(Copy, Clone)]
pub struct FrequencyResponseWithCoherence<T> {
    /// Angular frequency [rad/s].
    pub omega: T,
    /// `G(jω)`.
    pub value: Complex<T>,
    /// Coherence `|P_yu|^2 / (P_uu P_yy)`, in `[0, 1]`.
    pub coherence: T,
}

/// The response without the coherence.
impl<T> From<FrequencyResponseWithCoherence<T>> for FrequencyResponse<T> {
    fn from(val: FrequencyResponseWithCoherence<T>) -> Self {
        FrequencyResponse { omega: val.omega, value: val.value }
    }
}

/// Serialized flat as `[omega, re, im, coherence]`.
impl<T: Serialize> Serialize for FrequencyResponseWithCoherence<T> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut state = serializer.serialize_struct("FrequencyResponseWithCoherence", 4)?;
        state.serialize_field("omega", &self.omega)?;
        state.serialize_field("re", &self.value.re)?;
        state.serialize_field("im", &self.value.im)?;
        state.serialize_field("coherence", &self.coherence)?;
        state.end()
    }
}

/// Frequency responses with their coherence, one per frequency in order (e.g. the bins of
/// [`welch`](fft::welch)). Derefs to the slice of the responses (`len`, `iter`, indexing), and
/// converts into `Vec<FrequencyResponse<T>>` (all of them) with `into`. Serialized as the sequence.
#[derive(Clone, Serialize)]
#[serde(transparent)]
pub struct FrequencyResponsesWithCoherence<T>(pub Vec<FrequencyResponseWithCoherence<T>>);

impl<T: Float> FrequencyResponsesWithCoherence<T> {
    /// The responses that the data define: `ω > 0` and coherence at least `min_coherence` (e.g.
    /// `0.9`), in order, without the coherence. The DC bin is dropped whatever its coherence (that
    /// of [`welch`](fft::welch) is not the static gain: the mean of each segment is removed), and
    /// so are the unexcited bins (coherence zero). The result can be passed to the identification
    /// methods in
    /// [`system_identification::frequency_response`](crate::system_identification::frequency_response).
    ///
    /// The coherence measures the noise and the nonlinearity, not the excitation: without noise,
    /// the bins between the excited ones (window leakage) also have a coherence close to 1.
    pub fn coherent_bins(&self, min_coherence: T) -> Vec<FrequencyResponse<T>> {
        self.0
            .iter()
            .filter(|b| b.omega > T::zero() && b.coherence >= min_coherence)
            .map(|&b| b.into())
            .collect()
    }
}

impl<T> Deref for FrequencyResponsesWithCoherence<T> {
    type Target = [FrequencyResponseWithCoherence<T>];
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<T> IntoIterator for FrequencyResponsesWithCoherence<T> {
    type Item = FrequencyResponseWithCoherence<T>;
    type IntoIter = std::vec::IntoIter<FrequencyResponseWithCoherence<T>>;
    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

impl<'a, T> IntoIterator for &'a FrequencyResponsesWithCoherence<T> {
    type Item = &'a FrequencyResponseWithCoherence<T>;
    type IntoIter = std::slice::Iter<'a, FrequencyResponseWithCoherence<T>>;
    fn into_iter(self) -> Self::IntoIter {
        self.0.iter()
    }
}

impl<T> FromIterator<FrequencyResponseWithCoherence<T>> for FrequencyResponsesWithCoherence<T> {
    fn from_iter<I: IntoIterator<Item = FrequencyResponseWithCoherence<T>>>(iter: I) -> Self {
        Self(iter.into_iter().collect())
    }
}

impl<T> From<Vec<FrequencyResponseWithCoherence<T>>> for FrequencyResponsesWithCoherence<T> {
    fn from(val: Vec<FrequencyResponseWithCoherence<T>>) -> Self {
        Self(val)
    }
}

/// All the responses (with the DC bin, if any), without the coherence.
impl<T> From<FrequencyResponsesWithCoherence<T>> for Vec<FrequencyResponse<T>> {
    fn from(val: FrequencyResponsesWithCoherence<T>) -> Self {
        val.0.into_iter().map(FrequencyResponse::from).collect()
    }
}

/// Gain and phase \[rad\] at `frequency`, whose unit is `U` (`Hz` or `RadPerSec`).
#[derive(Copy, Clone, Serialize)]
pub struct FrequencyCharacteristics<T, U = Hz>{
    /// Frequency, in `U`.
    pub frequency: T,
    /// Gain, in dB or as the magnitude depending on how it was computed (`log_scale`).
    pub gain: T,
    /// Phase \[rad\], continuous along the frequency axis from `BodeDiagramPlotter` by default (see `BodeDiagramPlotter::unwrap_phase`), otherwise in `(-π, π]`.
    pub phase: T,
    #[serde(skip)]
    _unit: PhantomData<U>,
}

impl<T: Float, U> FrequencyCharacteristics<T, U> {
    /// All zeros.
    pub fn new() -> Self {
        Self { frequency: T::zero(), gain: T::zero(), phase: T::zero(), _unit: PhantomData }
    }
}

impl<T: Float + FloatConst> FrequencyCharacteristics<T, Hz> {
    /// Convert each response as `From<FrequencyResponse>` does (gain in dB, phase in `(-π, π]`, not unwrapped), with the frequency in Hz.
    pub fn from_responses(responses: &[FrequencyResponse<T>]) -> Vec<Self> {
        responses.iter().map(|&r| r.into()).collect()
    }
}

impl<T: Float + FloatConst> FrequencyCharacteristics<T, RadPerSec> {
    /// `from_responses` with the frequency in rad/s.
    pub fn from_responses_rad_per_sec(responses: &[FrequencyResponse<T>]) -> Vec<Self> {
        responses.iter().map(|&r| r.into()).collect()
    }
}

impl<T: Float, U> Default for FrequencyCharacteristics<T, U> {
    fn default() -> Self {
        Self::new()
    }
}

/// Gain in dB and phase in `(-π, π]`; `omega` [rad/s] is converted into `U`.
impl<T: Float + FloatConst, U: FrequencyUnit> From<FrequencyResponse<T>> for FrequencyCharacteristics<T, U> {
    fn from(response: FrequencyResponse<T>) -> Self {
        Self {
            frequency: U::from_rad_per_sec(response.omega),
            gain: T::from(20.0).unwrap() * response.value.norm().log10(),
            phase: response.value.arg(),
            _unit: PhantomData,
        }
    }
}
