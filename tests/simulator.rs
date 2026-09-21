#[test]
fn test_simulate() {

    use dsmc::logger::DataStorage;
    use dsmc::{BodeDiagramPlotter, TransferFunction, discretize::exact_discretize};


    let ts = 1e-4;
    let omega_c = 300.0;

    let iter = 20;
    for i in 0..=iter {
        let zeta = 0.0 + 2.0 / iter as f64 * i as f64;

        let system = TransferFunction::new(&[omega_c * omega_c], &[1.0, 2.0 * zeta * omega_c, omega_c * omega_c]);
        let mut system_d = exact_discretize::DiscretizedSystem::from_tf(&system, ts).unwrap();

        let bode_plotter = BodeDiagramPlotter::<f64>::new(0.0, 10000.0 / (2.0 * std::f64::consts::PI), 0.1, true);
        let frequency_response = bode_plotter.frequency_response_s(&system);

        let mut storage_bode = DataStorage::new(format!("./out/bode_2order/bode_zeta_{:.01}.csv", zeta), ',', false).unwrap();
        let mut storage_res = DataStorage::new(format!("./out/simulator_2order/response_zeta_{:.01}.csv", zeta), ',', false).unwrap();

        for res in frequency_response {
            storage_bode.add(&res).unwrap();
        }

        for i in 0..2000 {
            let t = i as f64 * ts;
            let r = if i > 400 { 1.0 } else { 0.0 };
            let y = system_d.update(&[r]).unwrap();

            storage_res.add(&[t, r, y[0]]).unwrap();
        }

        storage_bode.close().unwrap();
        storage_res.close().unwrap();
    }
}
