use std::ops::AddAssign;

use num_traits::{Float, FloatConst};

use crate::trajectory::TrajectoryProfile;

pub fn generate<T: Float + FloatConst + AddAssign>(distance: T, samples: usize) -> Vec<TrajectoryProfile<T>> {

    let mut t = T::zero();
    let dt = T::one() / T::from(samples - 1).unwrap();
    let pi = FloatConst::PI();

    let t0 = T::zero();
    let t1 = T::one() / T::from(8.0).unwrap();
    let t2 = T::from(3.0).unwrap() / T::from(8.0).unwrap();
    let t3 = T::one() - t2;
    let t4 = T::one() - t1;
    let t5 = T::one();

    let ts1 = t1 - t0;
    let ts2 = t2 - t1;
    let ts3 = t3 - t2;
    let ts5 = t5 - t4;

    let am = distance * (T::from(8.0).unwrap() * pi) / (pi + T::from(2.0).unwrap());

    let c1 = ts1 * FloatConst::FRAC_2_PI();
    let c2 = am * T::from(0.5).unwrap();
    let c3 = ts3 / pi;
    let c4 = am * T::from(0.5).unwrap();
    let c5 = ts5 * FloatConst::FRAC_2_PI();

    let v1 = c1 * am;
    let s1 = c1 * am * (t1 - c1);

    let v2 = am * ts2 + v1;
    let s2 = c2 * ts2 * ts2 + v1 * ts2 + s1;

    let v3 = v2;
    let s3 = distance - s2;

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
                s: c2 * tl * tl + v1 * tl + s1,
                v: am * tl + v1,
                a: am,
            }
        }
        else if t < t3 {
            let tl = t - t2;
            TrajectoryProfile {
                s: c3 * c3 * am * (T::one() - (tl / c3).cos()) + v2 * tl + s2,
                v: c3 * am * (tl / c3).sin() + v2,
                a: am * (tl / c3).cos(),
            }
        }
        else if t < t4 {
            let tl = t - t3;
            TrajectoryProfile {
                s: -c4 * tl * tl + v3 * tl + s3,
                v: - am * tl + v3,
                a: - am,
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
