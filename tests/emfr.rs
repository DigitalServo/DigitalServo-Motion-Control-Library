use std::ops::Sub;
use num_traits::Float;

use mclib::{
    TransferFunction,
    discretize::bilinear_transform,
    logger::DataStorage
};


#[derive(Copy, Clone)]
struct Motion<T> {
    x: T,
    v: T,
    a: T,
    f: T
}

impl<T: Float> Motion<T> {
    fn new() -> Self {
        Self { x: T::zero(), v: T::zero(), a: T::zero(), f: T::zero() }
    }
}

impl<T: Float> Sub for Motion<T> {
    type Output = Motion<T>;
    fn sub(self, rhs: Self) -> Self::Output {
        Self {
            x: self.x - rhs.x,
            v: self.v - rhs.v,
            a: self.a - rhs.a,
            f: self.f - rhs.f,
        }
    }
}

#[test]
fn emfr() {

    let g = 9.80665;

    let mut storage = DataStorage::new("./out/emfr.csv", ',', false).unwrap();

    let mut t = 0.0;
    let ts = 1e-4;

    let tp = 1e-7;
    let plant_iter = (ts/ tp).round() as usize;

    let weight_max = 0.6;

    let sample_m = 0.001;

    let pan_d = 0.00;
    let pan_m = 0.15;
    let mut pan = Motion::<f64>::new();

    let beam_k = 1000000.0;
    let beam_d = 0.01;

    let motor_m = 0.05;
    let mut motor = Motion::<f64>::new();

    // Controller
    let g_s = 10000.0;
    let kp: f64 = 160000.0;
    let kd: f64 = 2.0 * kp.sqrt();
    let controller = TransferFunction::new(&[kp + kd * g_s, kp * g_s], &[1.0, g_s]);
    let mut controller = bilinear_transform::DiscretizedSystem::new(controller, ts);

    let g_dob = 500.0;
    let q_filter_dob = TransferFunction::new(&[g_dob * g_dob], &[1.0, 2.0 * g_dob, g_dob * g_dob]);
    let mut q_filter_dob = bilinear_transform::DiscretizedSystem::new(q_filter_dob, ts);

    // By setting nominal weight larger than the actual weight, the DOB insert phase-lead-compensator on the open loop,
    // which increase stability margin.
    let m_dob = motor_m + weight_max;

    let g_rfob = 500.0;
    let q_filter_rfob = TransferFunction::new(&[g_rfob * g_rfob], &[1.0, 2.0 * g_rfob, g_rfob * g_rfob]);
    let mut q_filter_rfob = bilinear_transform::DiscretizedSystem::new(q_filter_rfob, ts);

    // let mut dist = 0.0;
    let mut load_m;

    let weight_offset = pan_m + motor_m;

    let mut weight_hat = 0.0;
    let mut weight_from_current = 0.0;
    let mut weight_complement = 0.0;

    let complement_gain = 0.7;

    for _ in 0..10000 {

        if t < 0.5 {
            load_m = pan_m;
        } else {
            load_m = pan_m + sample_m;
        }
        // } else if t < 0.8 {
        //     load_m = pan_m + sample_m;
        // } else if t < 0.81 {
        //     load_m = pan_m + sample_m + 1.0;
        // } else {
        //     load_m = pan_m;
        // }


        let u_fb = m_dob * controller.update(0.0 - motor.x);
        let u_dob = q_filter_dob.update(motor.f - m_dob * motor.a);

        motor.f = u_fb + u_dob;

        let d_est = q_filter_rfob.update(motor.f - (motor_m + pan_m) * motor.a);

        if t > 0.2 {
            weight_hat = (d_est / g) - weight_offset;
            weight_from_current = motor.f / g - weight_offset;
            weight_complement = complement_gain * weight_from_current + (1.0 - complement_gain) * weight_hat;
        }

        for _ in 0..plant_iter {

            let torsion = pan - motor;
            let torsion_force = beam_k * torsion.x + beam_d * torsion.v;

            motor.x += motor.v * tp;
            motor.v += motor.a * tp;
            motor.a = (motor.f + torsion_force - g * motor_m) / motor_m;

            pan.x += pan.v * tp;
            pan.v += pan.a * tp;
            pan.a = (- torsion_force - pan_d * pan.v - g * load_m) / load_m;

            t += tp;
        }

        storage.add(&[t, motor.x, pan.x, weight_from_current, weight_hat, weight_complement]).unwrap();
    }

    storage.close().unwrap();

}
