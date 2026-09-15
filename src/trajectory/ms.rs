use std::ops::AddAssign;

use num_traits::{Float, FloatConst};

use crate::trajectory::TrajectoryProfile;

pub fn generate<T: Float + FloatConst + AddAssign>(distance: T, samples: usize) -> Vec<TrajectoryProfile<T>> {

    let mut t = T::zero();
    let dt = T::one() / T::from(samples - 1).unwrap();
    let pi: T = FloatConst::PI();

    let t0 = T::zero();
    let t1 = T::one() / T::from(8.0).unwrap();
    let t2 = T::one() - t1;
    let t3 = T::one();

    // Maximum acceleration
    let am = distance * (pi * pi) / (T::one() + FloatConst::FRAC_PI_4());

    let ts1 = t1 - t0;
    let ts2 = t2 - t1;
    let ts3 = t3 - t2;

    let c1 = ts1 * FloatConst::FRAC_2_PI();
    let c2 = ts2 * FloatConst::FRAC_1_PI();
    let c3 = ts3 * FloatConst::FRAC_2_PI();

    let v1 = c1 * am;
    let s1 = c1 * am * (t1 - c1);

    let v2 = v1;
    let s2 = distance - s1;

    let s3 = distance;

    let mut trajectory = Vec::<TrajectoryProfile<T>>::with_capacity(samples);

    for _ in 0..samples {

        let data: TrajectoryProfile<T> = if t < t1 {
            let tl = t - t0;
            TrajectoryProfile {
                s: c1 * am * (tl - c1 * (tl / c1).sin()),
                v: c1 * am * (T::one() - (tl / c1).cos()),
                a: am * (tl / c1).sin(),
            }
        }
        else if t < t2 {
            let tl = t - t1;
            TrajectoryProfile {
                s: c2 * c2 * am * (T::one() - (tl / c2).cos()) + v1 * tl + s1,
                v: c2 * am * (tl / c2).sin() + v1,
                a: am * (tl / c2).cos(),
            }
        }
        else if t < t3 {
            let tl = t - t2;
            TrajectoryProfile {
                s: c3 * c3 * am * ((tl / c3).cos() - T::one()) + v2 * tl + s2,
                v: - c3 * am * (tl / c3).sin() + v2,
                a: - am * (tl / c3).cos(),
            }
        }
        else {
            TrajectoryProfile { s: s3, v: T::zero(), a: T::zero() }
        };

        trajectory.push(data);
        t += dt;
    }

    trajectory
}
