use mclib::{TransferFunction, discretize::exact_discretize::{DiscretizedSystem, LiftedDiscretizedSystem}, logger::DataStorage, trajectory};

#[test]
fn ptc() {

    let ts = 1e-4;

    let mut storage = DataStorage::new("./out/ptc.csv", ',', false).unwrap();

    let plant: TransferFunction<f64> = TransferFunction::<f64>::new(&[1.0], &[2.0e-4, 0.05, 0.0]);
    let mut model: DiscretizedSystem<f64> = DiscretizedSystem::from_tf(plant, ts).unwrap();
    let model_lifted: LiftedDiscretizedSystem<f64> = model.clone().try_into().unwrap();

    let rest_tlen = 0.02;
    let move_tlen = 0.05;
    let rest_samples = (rest_tlen / ts).round() as usize;
    let move_samples = (move_tlen / ts).round() as usize;
    let move_distance = 1.0;
    let trajectory_sin = trajectory::sin::generate(move_distance, move_samples);

    let mut r = Vec::<Vec<f64>>::with_capacity(rest_samples * 2 + move_samples);
    {
        for _ in 0..rest_samples {
            let p = vec![0.0, 0.0];
            r.push(p);
        }

        for i in 0..move_samples {
            let p = vec![trajectory_sin[i].s, trajectory_sin[i].v];
            r.push(p);
        }

        for _ in 0..rest_samples {
            let p = vec![move_distance, 0.0];
            r.push(p);
        }
    }

    let u = model_lifted.calculate_ptc_input(r.clone());

    for i in 0..u.len() {
        storage.add(&[ts * i as f64, r[i][0], model.output[0], (r[i][0] - model.output[0])]).unwrap();
        model.update(&[u[i]]).unwrap();
    }

    storage.close().unwrap();
}
