#[test]
fn test_smith_predictor_with_ptc() {

    use dsmc::{
        TransferFunction,
        discretize::{bilinear_transform, exact_discretize::DiscretizedSystem},
        feedforward::ptc::LiftedDiscretizedSystem,
        logger::DataStorage,
        signal::Delayer,
        trajectory::{self, ReferenceTrajectory},
    };

    let ts = 1e-4;

    let mut storage = DataStorage::new("./out/smith.csv", ',', false).unwrap();

    let j = 2.0e-4;
    let d = 0.02;

    let plant: TransferFunction<f64> = TransferFunction::<f64>::continuous(&[-0.01, 1.0], &[j, d, 0.0]);
    let mut plant_ssr: DiscretizedSystem<f64> = DiscretizedSystem::from_tf(plant, ts).unwrap();

    let model: TransferFunction<f64> = TransferFunction::<f64>::continuous(&[-0.01, 1.0], &[j * 1.0, d * 1.0, 0.0]);
    // model for disturbance observer
    let mut model_ssr: DiscretizedSystem<f64> = DiscretizedSystem::from_tf(model.clone(), ts).unwrap();
    let model_lifted: LiftedDiscretizedSystem<f64> = model_ssr.clone().try_into().unwrap();

    let over_sampling_rate: usize = 2;

    let plant_delay_time = 0.01;
    let plant_delay_sample = (plant_delay_time / ts).round() as usize;
    let mut plant_delayr = Delayer::<f64>::new(plant_delay_sample);

    let model_delay_time = 0.01;
    let model_delay_sample = (model_delay_time / ts).round() as usize;
    let mut model_delayr_1 = Delayer::<f64>::new(model_delay_sample);

    let g_q = 500.0;
    let q_filter = TransferFunction::continuous(&[(j * g_q * g_q), (d * g_q * g_q), 0.0], &[1.0, 2.0 * g_q, g_q * g_q]);
    let mut q_filter = bilinear_transform::DiscretizedSystem::new(q_filter, ts * over_sampling_rate as f64);

    let g_s = 1000.0;
    let kp: f64 = 900.0;
    let kd: f64 = 2.0 * kp.sqrt();
    let controller = TransferFunction::continuous(&[kp + kd * g_s, kp * g_s], &[1.0, g_s]);
    let mut controller = bilinear_transform::DiscretizedSystem::new(controller, ts * over_sampling_rate as f64);

    // Trajectory
    let rest_tlen = 0.5;
    let move_tlen = 0.5;
    let rest_samples = (rest_tlen / ts).round() as usize;
    let move_samples = (move_tlen / ts).round() as usize;
    let move_distance = 1.0;

    let trajectory = trajectory::SmoothPolynomial{smoothness: 0};
    let yd = trajectory.reference(move_distance, move_tlen, rest_tlen);
    let u_ptc = model_lifted.calculate_ptc_input_for_reference_output(&yd, rest_samples * 2 + move_samples).unwrap();

    let mut y = 0.0;
    let mut y_est = 0.0;
    let mut y_pred;

    let mut u_ff  = 0.0;
    let mut u_fb = 0.0;
    let mut u_dob = 0.0;
    let mut u_actual = 0.0;

    let mut dist;

    // Control with Preactuation
    let u_len = u_ptc.len() - (model_delay_sample + 1);
    for i in 0..u_len {

        let t = ts * i as f64;

        y_pred = yd.value(t);

        storage.add(&[t, y_pred, y, (y_pred - y), u_ff, u_actual]).unwrap();

        u_actual = {
            // multirate controller::feedforward
            u_ff = u_ptc[i + model_delay_sample + 1];
            // multirate controller::feedback
            if i % over_sampling_rate == 0 {
                u_fb = j * controller.update(y_pred - y);
                u_dob = q_filter.update(y_est - y);
            }
            u_ff + 1.0 * (u_fb + u_dob)
        };

        // disturbance
        dist = 0.01 + 0.001 * (2.0 * std::f64::consts::PI * 10.0 * t).sin();

        // plant
        plant_ssr.update(&[u_actual + 0.0 * dist]).unwrap();
        y = plant_delayr.output(plant_ssr.output[0]);

        // model (predictor)
        model_ssr.update(&[u_actual]).unwrap();
        y_est = model_delayr_1.output(model_ssr.output[0]);

    }

    storage.close().unwrap();
}
