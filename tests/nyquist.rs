use dsmc::logger::DataStorage;
use dsmc::{BodeDiagramPlotter, NyquistPlotter, TransferFunction};

#[test]
fn test_nyquist_plot() {

    let omega = 300.0;

    let kp = 3.0 * omega * omega;
    let kd = 3.0 * omega;
    let ki = omega* omega * omega;

    let iter = 10;
    for i in 0..=iter {
        let alpha = 0.8 + 0.4 / iter as f64 * i as f64;

        let mut storage_n = DataStorage::new(format!("./out/nyquist/nyquist_alpha_{:.02}.csv", alpha), ',', false).unwrap();
        let mut storage_b = DataStorage::new(format!("./out/bode/bode_alpha_{:.02}.csv", alpha), ',', false).unwrap();

        let plotter_n = NyquistPlotter::<f64>::new(10.0, 2000.0, 0.01);
        let plotter_b = BodeDiagramPlotter::<f64>::new(0.1, 1000.0, 0.01, true);

        let tf_main = TransferFunction::new(&[alpha * kd, alpha * kp, alpha * ki], &[1.0, 0.0]);
        let tf_pl = TransferFunction::new(&[1.0], &[1.0, 0.0, 0.0]);
        let tf = tf_main * tf_pl;
        {
            let responses = plotter_n.plot(&tf);
            for res in responses {
                storage_n.add(&[res.re, res.im]).unwrap();
            }
        }
        {
            let responses = plotter_b.frequency_response_s(&tf);
            for res in responses {
                storage_b.add(&res).unwrap();
            }
        }

        storage_n.close().unwrap();
        storage_b.close().unwrap();
    }

}
