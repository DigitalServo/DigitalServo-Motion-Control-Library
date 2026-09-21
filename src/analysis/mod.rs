use nalgebra::Complex;

mod bode_diagram;
mod nyquist;
mod statistics;
pub mod fft;

pub use bode_diagram::BodeDiagramPlotter;
pub use nyquist::NyquistPlotter;
pub use statistics::Statistics;

use num_traits::Float;
use serde::Serialize;

#[derive(Copy, Clone)]
pub struct FrequencyResponse<T> {
    pub omega: T,
    pub value: Complex<T>
}

#[derive(Copy, Clone, Serialize)]
pub struct FrequencyCharacteristics<T>{
    pub frequency: T,
    pub gain: T,
    pub phase: T,
}

impl<T: Float> FrequencyCharacteristics<T> {
    pub fn new() -> Self {
        Self { frequency: T::zero(), gain: T::zero(), phase: T::zero() }
    }
}
