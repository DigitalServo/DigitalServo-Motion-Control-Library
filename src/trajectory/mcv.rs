use std::ops::AddAssign;

use num_traits::{Float, FloatConst, ToPrimitive};

use crate::trajectory::TrajectoryProfile;

pub fn generate<S: ToPrimitive, T: Float + FloatConst + AddAssign>(percent: S, distance: T, samples: usize) -> Vec<TrajectoryProfile<T>> {

    let mut t = T::zero();
    let dt = T::one() / T::from(samples - 1).unwrap();
    let pi = FloatConst::PI();

    let proportion = T::from(percent).unwrap() / T::from(100.0).unwrap();

    let t0 = T::zero();
    let t1 = (T::one() - proportion) / T::from(8.0).unwrap();
    let t2 = t1 * T::from(4.0).unwrap();
    let t3 = T::one() - t2;
    let t4 = T::one() - t1;
    let t5 = T::one();

    let ts1 = t1 - t0;
    let ts2 = t2 - t1;
    let ts3 = t3 - t2;
    let ts4 = t4 - t3;
    let ts5 = t5 - t4;

    let am = distance * (pi * pi) / (T::from(8.0).unwrap() * t1 * (((T::from(8.0).unwrap() - T::from(6.0).unwrap() * pi) * t1) + pi));

    let c1 = ts1 * FloatConst::FRAC_2_PI();
    let c2 = ts2 * FloatConst::FRAC_2_PI();
    let c4 = ts4 * FloatConst::FRAC_2_PI();
    let c5 = ts5 * FloatConst::FRAC_2_PI();

    let v1 = c1 * am;
    let s1 = c1 * am * (t1 - c1);

    let v2 = c2 * am + v1;
    let s2 = (c2).powi(2) * am + v1 * ts2 + s1;

    let v3 = v2;
    let s3 = v2 * ts3 + s2;

    let v4 = v1;
    let s4 = distance - s1;

    let s5 = distance;

    let mut trajectory = Vec::<TrajectoryProfile<T>>::with_capacity(samples);

    for _ in 0..samples {

        let data = if t < t1 {
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
                s: c2.powi(2) * am * (T::one() - (tl / c2).cos()) + v1 * tl + s1,
                v: c2 * am * (tl / c2).sin() + v1,
                a: am * (tl / c2).cos(),
            }
        }
        else if t < t3 {
            let tl = t - t2;
            TrajectoryProfile {
                s: v2 * tl + s2,
                v: v2,
                a: T::zero(),
            }
        }
        else if t < t4 {
            let tl = t - t3;
            TrajectoryProfile {
                s: c4 * am * (c4 * (tl / c4).sin() - tl) + v3 * tl + s3,
                v: c4 * am * ((tl / c4).cos() - T::one()) + v3,
                a: - am * (tl / c4).sin(),
            }
        }
        else if t < t5 {
            let tl = t - t4;
            TrajectoryProfile {
                s: c5.powi(2) * am * ((tl/ c5).cos() - T::one()) + v4 * tl + s4,
                v: -c5 * am * (tl/ c5).sin() + v4,
                a: - am * (tl/ c5).cos(),
            }
        }
        else {
            TrajectoryProfile {
                s: s5,
                v: T::zero(),
                a: T::zero(),
            }
        };

        trajectory.push(data);
        t += dt;
    }

    trajectory
}
