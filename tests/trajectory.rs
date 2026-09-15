use mclib::{logger::DataStorage, trajectory};

#[test]
fn test_trajectory() {

    let samples = 800;
    let distance = 10.0;

    let trajectory_sin = trajectory::sin::generate(distance, samples);
    let trajectory_cycloid = trajectory::cycloid::generate(distance, samples);
    let trajectory_mt = trajectory::mt::generate(distance, samples);
    let trajectory_ms = trajectory::ms::generate(distance, samples);
    let trajectory_mcv20 = trajectory::mcv::generate(50, distance, samples);
    let trajectory_mcv80 = trajectory::mcv::generate(80, distance, samples);

    let mut storage = DataStorage::new("./out/trajectory.csv", ',', false).unwrap();

    for i in 0..samples {
        storage.add(&[
            i as f64,
            trajectory_ms[i].s,
            trajectory_mt[i].s,
            trajectory_mcv20[i].s,
            trajectory_sin[i].s,
            trajectory_cycloid[i].s,
            trajectory_mcv80[i].s,
        ]).unwrap();
    }

    storage.close().unwrap();
}
