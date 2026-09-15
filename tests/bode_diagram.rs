use mclib::discretize::bilinear_transform;
use mclib::logger::DataStorage;
use mclib::{BodeDiagramPlotter, TransferFunction};

#[test]
fn test_bode_plotter_s() {
    let mut storage = DataStorage::new("./out/bode_s.csv", ',', false).unwrap();

    let bode_plotter = BodeDiagramPlotter::<f64>::new(0.0, 1000.0, 0.01, true);

    // 1st order LPF
    let g = 2.0 * std::f64::consts::PI * 10.0;
    let tf_s = TransferFunction::new(&[g * g], &[1.0, 0.01 * g, g * g]);
    let responses = bode_plotter.frequency_response_s(&tf_s);

    for res in responses {
        storage.add(&res).unwrap();
    }

    storage.close().unwrap()
}

#[test]
fn test_bode_plotter_z() {
    let mut storage = DataStorage::new("./out/bode_z.csv", ',', false).unwrap();

    let ts = 1e-4;

    let bode_plotter = BodeDiagramPlotter::<f64>::new(0.0, 1000.0,  0.01, true);

    // 1st order LPF
    let g = 2.0 * std::f64::consts::PI * 10.0;
    // let tf_z = TransferFunction::new(&[g * ts, g * ts], &[2.0 + g * ts, -2.0 + g * ts]);

    let tf_s = TransferFunction::new(&[g * g], &[1.0, 0.01 * g, g * g]);
    let tf_z = bilinear_transform::discretize(tf_s, ts);

    let responses = bode_plotter.frequency_response_z(&tf_z, ts);

    for res in responses {
        storage.add(&res).unwrap();
    }

    storage.close().unwrap()
}
