//! Excitation signals for identification experiments.
//!
//! - periodic: `multisine`, `shaped_multisine`, `flat_output_multisine` (see `multisine.rs`)
//! - swept sine: `chirp`, `shaped_chirp` (see `chirp.rs`)
//! - pseudo-random binary: `MSequence`, `ShapedMSequence` (see `m_sequence.rs`)

use num_traits::Float;
use thiserror::Error;

use crate::sampling::DurationError;

mod chirp;
mod m_sequence;
mod multisine;

pub use chirp::{chirp, shaped_chirp};
pub use m_sequence::{MSequence, ShapedMSequence};
pub use multisine::{flat_output_multisine, multisine, shaped_multisine};

/// Error of a duration `duration / ts` that is not a whole number of samples.
pub(super) fn duration_error<T: Float>(kind: DurationError, duration: T, ts: T) -> ExcitationError {
    let (duration, ts) = (duration.to_f64().unwrap_or(f64::NAN), ts.to_f64().unwrap_or(f64::NAN));
    match kind {
        DurationError::Invalid => ExcitationError::InvalidDuration { duration, ts },
        DurationError::Fractional => ExcitationError::FractionalDuration { duration, ts },
    }
}

/// Invalid arguments of the excitation signals.
#[derive(Clone, Debug, Error, PartialEq)]
pub enum ExcitationError {
    #[error("duration {duration} s is negative or not finite, or sampling period {ts} s not positive and finite")]
    InvalidDuration { duration: f64, ts: f64 },
    #[error("duration {duration} s is not a whole number of sampling periods {ts} s")]
    FractionalDuration { duration: f64, ts: f64 },
    #[error("fundamental frequency {fundamental_frequency} Hz is not positive and finite, or sampling period {ts} s not positive and finite")]
    InvalidFundamental { fundamental_frequency: f64, ts: f64 },
    #[error("the period of the fundamental frequency {fundamental_frequency} Hz is not a whole number of sampling periods {ts} s")]
    FractionalPeriod { fundamental_frequency: f64, ts: f64 },
    #[error("harmonic {harmonic} is not within 1 <= h < N / 2 (N = {samples_per_period} samples per period)")]
    InvalidHarmonic { harmonic: usize, samples_per_period: usize },
    #[error("harmonic {harmonic} is given more than once")]
    DuplicateHarmonic { harmonic: usize },
    #[error("the amplitude of harmonic {harmonic} is zero or not finite")]
    InvalidAmplitude { harmonic: usize },
    #[error("no harmonics or samples, or their total power is zero or not finite")]
    NoPower,
    #[error("order {order} of the M-sequence is not within 2 ..= 32")]
    InvalidOrder { order: usize },
    #[error("clock {clock} is zero, or the period of (2^{order} - 1) bits of {clock} samples overflows usize")]
    InvalidClock { clock: usize, order: usize },
}
