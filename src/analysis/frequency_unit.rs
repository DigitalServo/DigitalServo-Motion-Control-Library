use num_traits::{Float, FloatConst};

/// Marker for frequencies in hertz.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Hz;

/// Marker for angular frequencies in rad/s.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RadPerSec;

/// Unit of a frequency axis: `Hz` or `RadPerSec`.
pub trait FrequencyUnit: Clone + Copy + std::fmt::Debug {
    /// Convert a frequency in this unit into an angular frequency [rad/s].
    fn to_rad_per_sec<T: Float + FloatConst>(frequency: T) -> T;
}

impl FrequencyUnit for Hz {
    fn to_rad_per_sec<T: Float + FloatConst>(frequency: T) -> T {
        T::from(2.0).unwrap() * T::PI() * frequency
    }
}

impl FrequencyUnit for RadPerSec {
    fn to_rad_per_sec<T: Float + FloatConst>(frequency: T) -> T {
        frequency
    }
}
