use std::ops::AddAssign;

use num_traits::{Float, FloatConst};

use crate::trajectory::TrajectoryProfile;

pub fn generate<T: Float + FloatConst + AddAssign>(distance: T, samples: usize) -> Vec<TrajectoryProfile<T>> {

    let mut t = T::zero();
    let dt = T::one() / T::from(samples - 1).unwrap();

    let omega: T = T::from(2.0).unwrap() * FloatConst::PI();
    let r: T = T::one() / omega;

    let mut trajectory = Vec::<TrajectoryProfile<T>>::with_capacity(samples);

    for _ in 0..samples {
        let s = distance * (t - r * (omega * t).sin());
        let v = distance * (T::one() - r * omega * (omega * t).cos());
        let a = distance * r * omega * omega * (omega * t).sin();
        trajectory.push(TrajectoryProfile { s, v, a });
        t += dt;
    }

    trajectory
}
