//! System analysis tools.

use std::marker::PhantomData;

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

use num_traits::Float;
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

impl<T: Float, U> Default for FrequencyCharacteristics<T, U> {
    fn default() -> Self {
        Self::new()
    }
}
