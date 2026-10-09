//! Position of the reference trajectories (`TrajectoryKind`) over 800 samples, written to
//! `out/trajectory.csv`.
//!
//! `cargo run --example trajectory`

use dsmc::logger::DataStorage;
use dsmc::trajectory::{Trajectory, TrajectoryKind};

fn main() {

    let samples = 800;
    let distance = 10.0;

    let trajectory_sin = TrajectoryKind::Sin.generate(distance, samples);
    let trajectory_cycloid = TrajectoryKind::Cycloid.generate(distance, samples);
    let trajectory_mt = TrajectoryKind::ModifiedTrapezoid.generate(distance, samples);
    let trajectory_ms = TrajectoryKind::ModifiedSine.generate(distance, samples);
    let trajectory_mcv20 = TrajectoryKind::ModifiedConstantVelocity { constant_velocity_percent: 50.0 }.generate(distance, samples);
    let trajectory_mcv80 = TrajectoryKind::ModifiedConstantVelocity { constant_velocity_percent: 80.0 }.generate(distance, samples);
    let trajectory_poly = TrajectoryKind::SmoothPolynomial { smoothness: 4 }.generate(distance, samples);

    let mut storage = DataStorage::new("./out/trajectory.csv").unwrap();

    for i in 0..samples {
        storage.add(&[
            i as f64,
            trajectory_ms[i].s,
            trajectory_mt[i].s,
            trajectory_mcv20[i].s,
            trajectory_sin[i].s,
            trajectory_cycloid[i].s,
            trajectory_mcv80[i].s,
            trajectory_poly[i].s
        ]).unwrap();
    }

    storage.close().unwrap();
}
