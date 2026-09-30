use dsmc::discretize::bilinear_transform;
use dsmc::logger::DataStorage;
use dsmc::{BodeDiagramPlotter, tf};

#[test]
fn test_bode_plotter_s() {
    let mut storage = DataStorage::new("./out/bode_s.csv", ',', false).unwrap();

    let bode_plotter = BodeDiagramPlotter::<f64>::new(0.0, 1000.0, 0.01, true);

    // 1st order LPF
    let g = 2.0 * std::f64::consts::PI * 10.0;
    let k1 = 0.01 * g;
    let k2 = g * g;
    let tf_s = tf!("{k2} / (s^2 + {k1}s + {k2})");
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


    let k1 = 0.01 * g;
    let k2 = g * g;
    let tf_s = tf!("{k2} / (s^2 + {k1}s + {k2})");
    let tf_z = bilinear_transform::discretize(tf_s, ts);

    // let tf_z = tf!("({}z + {})  / ({}z + {})", g * ts, g * ts, 2.0 + g * ts, -2.0 + g * ts);

    let responses = bode_plotter.frequency_response_z(&tf_z, ts);

    for res in responses {
        storage.add(&res).unwrap();
    }

    storage.close().unwrap()
}
