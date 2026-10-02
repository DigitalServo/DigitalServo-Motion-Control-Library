//! Cycloid profile.

use num_traits::{Float, FloatConst};

use crate::trajectory::{Trajectory, TrajectoryProfile};

/// Cycloid profile: `s = distance (x - sin(2π x) / (2π))`, with zero acceleration at both ends.
#[derive(Clone, Copy, Debug)]
pub struct Cycloid;

impl<T: Float + FloatConst> Trajectory<T> for Cycloid {
    fn profile(&self, distance: T, x: T) -> TrajectoryProfile<T> {
        if x < T::zero() {
            return TrajectoryProfile::rest(T::zero());
        }
        if x > T::one() {
            return TrajectoryProfile::rest(distance);
        }
        let omega: T = T::from(2.0).unwrap() * FloatConst::PI();
        let r: T = T::one() / omega;
        TrajectoryProfile {
            s: distance * (x - r * (omega * x).sin()),
            v: distance * (T::one() - r * omega * (omega * x).cos()),
            a: distance * r * omega * omega * (omega * x).sin(),
        }
    }
}
