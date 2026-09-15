use std::ops::AddAssign;

use num_traits::{Float, FloatConst};

use crate::trajectory::TrajectoryProfile;

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
