//! Step responses of a second-order system discretized by a zero-order hold, as a state-space model
//! (`out/zoh_ssr_out.csv`) and as a transfer function (`out/zoh_tf_out.csv`).
//!
//! `cargo run --example zoh`

use dsmc::{DiscreteSystem, StateSpace, TransferFunction, discretize::Zoh, logger::DataStorage};
use nalgebra::dmatrix;

fn state_space() {

    let ts = 1.0e-4;
    let mut storage = DataStorage::new("./out/zoh_ssr_out.csv").unwrap();

    let g = 10.0;
    let system = StateSpace::new(
        dmatrix![0.0, 1.0; -g * g, -2.0 * g],
        dmatrix![0.0; 1.0],
        dmatrix![g * g, 0.0],
        dmatrix![0.0],
    ).unwrap();
    let mut system = DiscreteSystem::from(system.discretize(Zoh, ts).unwrap());

    let mut t = 0.0;
    for _ in 0..20000 {
        let x = if t < 0.2 { 0.0 } else { 1.0 };
        let y = system.update(&[x]).unwrap();
        storage.add(&[t, x, y[0]]).unwrap();

        t += ts;
    }

    storage.close().unwrap();
}

fn transfer_function() {

    let ts = 1.0e-4;
    let mut storage = DataStorage::new("./out/zoh_tf_out.csv").unwrap();

    let g = 10.0;
    let system = TransferFunction::continuous(&[g * g], &[1.0, 2.0 * g, g * g]);
    let mut system = DiscreteSystem::try_from(&system.discretize(Zoh, ts).unwrap()).unwrap();

    let mut t = 0.0;
    for _ in 0..20000 {
        let x = if t < 0.2 { 0.0 } else { 1.0 };
        let y = system.update(x);
        storage.add(&[t, x, y]).unwrap();

        t += ts;
    }

    storage.close().unwrap();
}

fn main() {
    state_space();
    transfer_function();
}
