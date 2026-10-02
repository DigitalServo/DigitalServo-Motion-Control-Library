//! Half-cosine (harmonic) profile.

use std::ops::AddAssign;

use num_traits::{Float, FloatConst};

use crate::laplace_transform::DelayedRationalSum;
use crate::trajectory::{ReferenceTrajectory, Trajectory, TrajectoryProfile};
use crate::{Polynomial, TransferFunction};

/// Half-cosine (harmonic) profile: `s = distance / 2 (1 - cos(π x))`. Besides the normalized profile
/// ([`Trajectory`]), it has an exact Laplace-domain form on the time axis ([`ReferenceTrajectory`]).
#[derive(Clone, Copy, Debug)]
pub struct Sin;

impl<T: Float + FloatConst> Trajectory<T> for Sin {
    fn profile(&self, distance: T, x: T) -> TrajectoryProfile<T> {
        if x < T::zero() {
            return TrajectoryProfile::rest(T::zero());
        }
        if x > T::one() {
            return TrajectoryProfile::rest(distance);
        }
        let pi: T = FloatConst::PI();
        let gain = distance * T::from(0.5).unwrap();
        TrajectoryProfile {
            s: gain * (T::one() - (pi * x).cos()),
            v: gain * pi * (pi * x).sin(),
            a: gain * pi * pi * (pi * x).cos(),
        }
    }
}

impl<T: Float + FloatConst + AddAssign> ReferenceTrajectory<T> for Sin {
    type Reference = DelayedRationalSum<T>;

    /// Exact Laplace-domain form (see `laplace_transform`).
    fn reference(&self, distance: T, duration: T, start: T) -> DelayedRationalSum<T> {
        laplace_transform(distance, duration, start)
    }
}

/// Laplace-domain form of the same profile: a move of `distance` in `duration` \[s\] starting at
/// `start` \[s\] (0 before, `distance` after). With `ω = π / duration`,
/// ```text
/// y(t) = distance / 2 (1 - cos(ω (t - start)))      (start <= t <= start + duration)
/// Y(s) = distance / 2 · ω² / (s (s² + ω²)) · (e^(-s start) + e^(-s (start + duration)))
/// ```
/// (after the move, `1 - cos` restarted at `start + duration` adds up to the constant `distance`).
/// `Sin.generate(distance, samples)` corresponds to `duration = (samples - 1) ts`.
pub fn laplace_transform<T: Float + FloatConst + AddAssign>(distance: T, duration: T, start: T) -> DelayedRationalSum<T> {
    let omega = T::PI() / duration;
    let half = distance * T::from(0.5).unwrap();
    let rational = TransferFunction::from_polynomials(
        Polynomial(vec![half * omega * omega]),
        Polynomial(vec![T::one(), T::zero(), omega * omega, T::zero()]),
    );
    let mut signal = DelayedRationalSum::new();
    signal.push(start, rational.clone()).push(start + duration, rational);
    signal
}
