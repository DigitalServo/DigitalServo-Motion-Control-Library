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
    let ssr_z = discretize_ssr(&system, ts).unwrap();

    // A has the double eigenvalue λ = -g, so e^{At} = e^{λt}(I + t(A - λI)) with
    // A - λI = [[g, 1], [-g^2, -g]]. Then
    //   A_d = e^{λT} [[1 + gT, T], [-g^2 T, 1 - gT]]
    //   B_d = ∫_0^T e^{Aτ} B dτ = [I1, I0 - g I1],  I0 = ∫ e^{λτ}, I1 = ∫ τ e^{λτ}
    let lambda: f64 = -g;
    let e = (lambda * ts).exp();
    let i0 = (e - 1.0) / lambda;
    let i1 = e * (ts / lambda - 1.0 / (lambda * lambda)) + 1.0 / (lambda * lambda);
    let expected_a = dmatrix![e * (1.0 + g * ts), e * ts; -g * g * ts * e, e * (1.0 - g * ts)];
    let expected_b = dmatrix![i1; i0 - g * i1];

    assert!((&ssr_z.a - &expected_a).abs().max() < 1e-12, "A_d = {} expected {}", ssr_z.a, expected_a);
    assert!((&ssr_z.b - &expected_b).abs().max() < 1e-15, "B_d = {} expected {}", ssr_z.b, expected_b);
    // C and D are unchanged by discretization.
    assert_eq!(ssr_z.c, system.c);
    assert_eq!(ssr_z.d, system.d);
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
    let system = TransferFunction::continuous(&[g * g], &[1.0, 2.0 * g, g * g]);
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
