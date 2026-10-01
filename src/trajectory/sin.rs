use std::ops::AddAssign;

use num_traits::{Float, FloatConst};

use crate::trajectory::TrajectoryProfile;
use crate::{LaplaceSignal, Polynomial, TransferFunction};

pub fn generate<T: Float + FloatConst + AddAssign>(distance: T, samples: usize) -> Vec<TrajectoryProfile<T>> {

    let mut t = T::zero();
    let dt: T = T::one() / T::from(samples - 1).unwrap();
    let pi: T = FloatConst::PI();

    let gain = distance * T::from(0.5).unwrap();

    let mut trajectory = Vec::<TrajectoryProfile<T>>::with_capacity(samples);

    for _ in 0..samples {
        let s = gain * (T::one() - (pi * t).cos());
        let v = gain * pi * (pi * t).sin();
        let a = gain * pi * pi * (pi * t).cos();
        trajectory.push(TrajectoryProfile { s, v, a });
        t += dt;
    }

    trajectory
}

/// Laplace-domain form of the same profile: a move of `distance` in `duration` [s] starting at
/// `start` [s] (0 before, `distance` after). With `ω = π / duration`,
/// ```text
/// y(t) = distance / 2 (1 - cos(ω (t - start)))      (start <= t <= start + duration)
/// Y(s) = distance / 2 · ω² / (s (s² + ω²)) · (e^(-s start) + e^(-s (start + duration)))
/// ```
/// (after the move, `1 - cos` restarted at `start + duration` adds up to the constant `distance`).
/// `generate(distance, samples)` corresponds to `duration = (samples - 1) ts`.
pub fn laplace<T: Float + FloatConst + AddAssign>(distance: T, duration: T, start: T) -> LaplaceSignal<T> {
    let omega = T::PI() / duration;
    let half = distance * T::from(0.5).unwrap();
    let rational = TransferFunction::from_polynomials(
        Polynomial(vec![half * omega * omega]),
        Polynomial(vec![T::one(), T::zero(), omega * omega, T::zero()]),
    );
    let mut signal = LaplaceSignal::new();
    signal.push(start, rational.clone()).push(start + duration, rational);
    signal
}
