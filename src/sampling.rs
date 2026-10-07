//! Durations in seconds as whole numbers of samples.

use num_traits::Float;

/// Why a duration is not a whole number of samples.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum DurationError {
    /// The duration is negative or not finite, or the sampling period not positive and finite.
    Invalid,
    /// `duration / ts` is not a whole number.
    Fractional,
}

/// Number of samples `duration / ts` in `duration` \[s\] (at least 0) with the sampling period `ts`
/// \[s\], if it is a whole number. Relative tolerance `max(1e-9, 4 eps)`: the rounding of
/// `duration`, `ts` and of the division (e.g. `0.05f32 / 1e-4 = 500.00003`).
pub(crate) fn whole_samples<T: Float>(duration: T, ts: T) -> Result<usize, DurationError> {
    if !(duration >= T::zero() && duration.is_finite() && ts > T::zero() && ts.is_finite()) {
        return Err(DurationError::Invalid);
    }
    let samples = duration / ts;
    if !samples.is_finite() {
        return Err(DurationError::Invalid);
    }
    let whole = samples.round();
    let tolerance = T::from(1e-9).unwrap().max(T::from(4).unwrap() * T::epsilon());
    if (samples - whole).abs() > tolerance * samples.max(T::one()) {
        return Err(DurationError::Fractional);
    }
    whole.to_usize().ok_or(DurationError::Invalid)
}

/// Number of samples `1 / (fundamental_frequency ts)` in a period of the fundamental frequency
/// \[Hz\], if it is a whole number (the sampling frequency a whole multiple of the fundamental),
/// as `whole_samples`.
pub(crate) fn samples_per_period<T: Float>(fundamental_frequency: T, ts: T) -> Result<usize, DurationError> {
    if !(fundamental_frequency > T::zero() && fundamental_frequency.is_finite()) {
        return Err(DurationError::Invalid);
    }
    whole_samples(T::one() / fundamental_frequency, ts)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn whole_numbers_of_samples() {
        assert_eq!(whole_samples(1.0, 1e-3), Ok(1000));
        assert_eq!(whole_samples(0.1, 1e-3), Ok(100)); // 100.00000000000001
        assert_eq!(whole_samples(0.0, 1e-3), Ok(0));
        assert_eq!(whole_samples(1.0f32, 1e-4), Ok(10000));
        assert_eq!(whole_samples(0.05f32, 1e-4), Ok(500)); // 500.00003 in f32
        assert_eq!(whole_samples(1.0005, 1e-3), Err(DurationError::Fractional));
        for (duration, ts) in [(-1.0, 1e-3), (1.0, 0.0), (1.0, -1e-3), (f64::NAN, 1e-3), (f64::INFINITY, 1e-3), (1.0, f64::NAN), (1e300, 1e-300)] {
            assert_eq!(whole_samples(duration, ts), Err(DurationError::Invalid), "{duration}, {ts}");
        }
    }

    #[test]
    fn periods_of_the_fundamental() {
        assert_eq!(samples_per_period(1.0, 1e-3), Ok(1000));
        assert_eq!(samples_per_period(10.0, 1e-4), Ok(1000));
        assert_eq!(samples_per_period(0.1, 1e-3), Ok(10000));
        assert_eq!(samples_per_period(20.0f32, 1e-4), Ok(500));
        assert_eq!(samples_per_period(3.0, 1e-3), Err(DurationError::Fractional)); // 333.3 samples
        for f in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            assert_eq!(samples_per_period(f, 1e-3), Err(DurationError::Invalid), "{f}");
        }
    }
}
