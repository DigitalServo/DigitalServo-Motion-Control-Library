//! Second-order low-pass filters discretized by Tustin: the Bode diagrams of the continuous and
//! the discrete filter (`out/bilinear_bode_s.csv`, `out/bilinear_bode_z.csv`) and a step response
//! (`out/filter_out.csv`).
//!
//! `cargo run --example bilinear_transform`

use dsmc::{BodeDiagramPlotter, DiscreteSystem, TransferFunction, discretize::Tustin, logger::DataStorage};

fn bode() {

    let ts = 1e-4;

    let g = 2.0 * std::f64::consts::PI * 10.0;

    let tf_s = TransferFunction::continuous(&[g * g],  &[1.0, 0.01 * g, g * g]);
    let tf_z = tf_s.discretize(Tustin, ts).unwrap();

    let bode_plotter = BodeDiagramPlotter::<f64>::new(0.0, 1000.0, 0.01, true);

    let mut storage_s = DataStorage::new("./out/bilinear_bode_s.csv").unwrap();
    let mut storage_z = DataStorage::new("./out/bilinear_bode_z.csv").unwrap();

    for res in bode_plotter.frequency_response_s(&tf_s) {
        storage_s.add(&res).unwrap();
    };
    for res in bode_plotter.frequency_response_z(&tf_z, ts) {
        storage_z.add(&res).unwrap();
    };
}

fn step_response() {

    let ts = 1e-4;
    let mut storage = DataStorage::new("./out/filter_out.csv").unwrap();

    let g = 10.0;

    let tf = TransferFunction::continuous(&[g * g], &[1.0, 2.0 * g, g * g]);

    let mut filter = DiscreteSystem::try_from(&tf.discretize(Tustin, ts).unwrap()).unwrap();

    let mut t = 0.0;
    for _ in 0..20000 {
        let x = if t < 0.2 { 0.0 } else { 1.0 };
        let y = filter.update(x);
        storage.add(&[t, x, y]).unwrap();

        t += ts;
    }

    storage.close().unwrap();
}

fn main() {
    bode();
    step_response();
}
