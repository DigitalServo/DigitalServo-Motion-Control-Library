use dsmc::{
    StateSpace, TransferFunction, discretize::exact_discretize::{DiscretizedSystem, discretize_ssr}, logger::DataStorage
};
use nalgebra::dmatrix;

#[test]
fn test_exact_discretize_ssr() {

    let ts = 1.0e-4;

    let g = 10.0;
    let system = StateSpace::new(
        dmatrix![0.0, 1.0; -g * g, -2.0 * g],
        dmatrix![0.0; 1.0],
        dmatrix![g * g, 0.0],
        dmatrix![0.0],
    ).unwrap();
    let ssr_z = discretize_ssr(&system, ts);

    println!("{:#?}", ssr_z);
}

#[test]
fn test_exact_discretize_system_ssr() {

    let ts = 1.0e-4;
    let mut storage = DataStorage::new("./out/exact_discretized_ssr_out.csv", ',', false).unwrap();

    let g = 10.0;
    let system = StateSpace::new(
        dmatrix![0.0, 1.0; -g * g, -2.0 * g],
        dmatrix![0.0; 1.0],
        dmatrix![g * g, 0.0],
        dmatrix![0.0],
    ).unwrap();
    let mut system = DiscretizedSystem::from_ssr(system, ts).unwrap();

    let mut t = 0.0;
    for _ in 0..20000 {
        let x = if t < 0.2 { 0.0 } else { 1.0 };
        let y = system.update(&[x]).unwrap();
        storage.add(&[t, x, y[0]]).unwrap();

        t += ts;
    }

    storage.close().unwrap();
}


#[test]
fn test_exact_discretize_system_tf() {

    let ts = 1.0e-4;
    let mut storage = DataStorage::new("./out/exact_discretized_tf_out.csv", ',', false).unwrap();

    let g = 10.0;
    let system = TransferFunction::new(&[g * g], &[1.0, 2.0 * g, g * g]);
    let mut system = DiscretizedSystem::from_tf(system, ts).unwrap();

    let mut t = 0.0;
    for _ in 0..20000 {
        let x = if t < 0.2 { 0.0 } else { 1.0 };
        let y = system.update(&[x]).unwrap();
        storage.add(&[t, x, y[0]]).unwrap();

        t += ts;
    }

    storage.close().unwrap();
}
