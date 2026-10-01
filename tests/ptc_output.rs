use dsmc::discretize::exact_discretize::{DiscretizedSystem, LiftedDiscretizedSystem, PtcError};
use dsmc::{tf, trajectory, PiecewisePolynomial, StableInverseError, StateSpace, TransferFunction};
use nalgebra::DMatrix;

const TS: f64 = 1e-4;
const REST: f64 = 0.02;
const DURATION: f64 = 0.05;

/// Relative degree 4 with a zero at +1000 rad/s.
fn plant() -> TransferFunction<f64> {
    tf!("(1 - 0.001s) / (s (0.005s + 1) (0.001s + 1) (0.0005s + 1) (0.0002s + 1))")
}

fn y_d() -> PiecewisePolynomial<f64> {
    trajectory::polynomial::piecewise(1.0, DURATION, REST, 4)
}

/// Simulate `u` on `model` and return the largest |y - y_d| at frame instants.
fn max_frame_error(mut model: DiscretizedSystem<f64>, u: &[f64], order: usize) -> f64 {
    let y = y_d();
    let mut max_error: f64 = 0.0;
    for (i, &ui) in u.iter().enumerate() {
        model.update(&[ui]).unwrap();
        if (i + 1) % order == 0 {
            max_error = max_error.max((model.output[0] - y.value((i + 1) as f64 * TS)).abs());
        }
    }
    max_error
}

fn samples() -> usize {
    ((2.0 * REST + DURATION) / TS).round() as usize
}

#[test]
fn canonical_realization() {
    let model = DiscretizedSystem::from_tf(plant(), TS).unwrap();
    let lifted: LiftedDiscretizedSystem<f64> = model.clone().try_into().unwrap();
    let u = lifted.calculate_ptc_input_from_reference_output(&y_d(), 0.0, samples()).unwrap();
    assert_eq!(u.len(), samples().div_ceil(5) * 5);

    // Same input as building the state reference by hand.
    let r = plant().state_reference(&y_d()).unwrap().sample(0.0, TS, u.len() + 1);
    let u_state = lifted.calculate_ptc_input_from_reference_state(r);
    // The plant is re-derived from (A, B, C), so the inputs agree up to that round-off
    // (the fast lags filter it out of the output).
    for (a, b) in u.iter().zip(&u_state) {
        assert!((a - b).abs() <= 1e-5 * b.abs().max(1.0), "{a} != {b}");
    }
    let error = max_frame_error(model, &u, 5);
    assert!(error < 1e-9, "max tracking error at frames: {error:e}");
}

#[test]
fn arbitrary_realization() {
    // Same plant as a cascade (not the canonical form):
    // u -> 1/(0.0002s+1) -> x5 -> 1/(0.0005s+1) -> x4 -> 1/(0.001s+1) -> x3 -> 1/(0.005s+1) -> x2 -> 1/s -> x1,
    // y = (1 - 0.001s) x1 = x1 - 0.001 x2
    let lag = |tau: f64| 1.0 / tau;
    #[rustfmt::skip]
    let a = DMatrix::from_row_slice(5, 5, &[
        0.0, 1.0,          0.0,          0.0,           0.0,
        0.0, -lag(0.005),  lag(0.005),   0.0,           0.0,
        0.0, 0.0,          -lag(0.001),  lag(0.001),    0.0,
        0.0, 0.0,          0.0,          -lag(0.0005),  lag(0.0005),
        0.0, 0.0,          0.0,          0.0,           -lag(0.0002),
    ]);
    let b = DMatrix::from_row_slice(5, 1, &[0.0, 0.0, 0.0, 0.0, lag(0.0002)]);
    let c = DMatrix::from_row_slice(1, 5, &[1.0, -0.001, 0.0, 0.0, 0.0]);
    let ssr = StateSpace::new(a, b, c, DMatrix::zeros(1, 1)).unwrap();

    let model = DiscretizedSystem::from_ssr(&ssr, TS).unwrap();
    let lifted: LiftedDiscretizedSystem<f64> = model.clone().try_into().unwrap();
    let u = lifted.calculate_ptc_input_from_reference_output(&y_d(), 0.0, samples()).unwrap();
    let error = max_frame_error(model, &u, 5);
    assert!(error < 1e-9, "max tracking error at frames: {error:e}");

    // The state reference is in the cascade's coordinates: x1 = ξ (position), x2 = ξ'.
    let x_d = lifted.reference_state(&y_d()).unwrap();
    let canonical = plant().state_reference(&y_d()).unwrap();
    for t in [0.0, REST, REST + 0.5 * DURATION, 1.0] {
        let x = x_d(t);
        let xc = canonical.state(t);
        assert!((x[0] - xc[0]).abs() < 1e-9 && (x[1] - xc[1]).abs() < 1e-6, "{x} vs {xc:?}");
    }
}

#[test]
fn errors() {
    // Zero on the imaginary axis: no stable inverse.
    let model = DiscretizedSystem::from_tf(tf!("(s^2 + 4) / (s + 1)^3"), TS).unwrap();
    let lifted: LiftedDiscretizedSystem<f64> = model.try_into().unwrap();
    assert!(matches!(
        lifted.calculate_ptc_input_from_reference_output(&y_d(), 0.0, 100),
        Err(PtcError::StableInverse(StableInverseError::PoleOnImaginaryAxis { .. }))
    ));

    // Direct feedthrough
    let ssr = StateSpace::new(
        DMatrix::from_row_slice(1, 1, &[-1.0]),
        DMatrix::from_row_slice(1, 1, &[1.0]),
        DMatrix::from_row_slice(1, 1, &[1.0]),
        DMatrix::from_row_slice(1, 1, &[0.5]),
    )
    .unwrap();
    let lifted: LiftedDiscretizedSystem<f64> = DiscretizedSystem::from_ssr(&ssr, TS).unwrap().try_into().unwrap();
    assert_eq!(
        lifted.calculate_ptc_input_from_reference_output(&y_d(), 0.0, 100).unwrap_err(),
        PtcError::Feedthrough
    );
}


#[test]
fn ptc_for_reference_state() {
    use dsmc::logger::DataStorage;

    let mut storage = DataStorage::new("./out/ptc.csv", ',', false).unwrap();

    let plant: TransferFunction<f64> = TransferFunction::<f64>::continuous(&[1.0], &[2.0e-4, 0.05, 0.0]);
    let mut model: DiscretizedSystem<f64> = DiscretizedSystem::from_tf(plant, TS).unwrap();
    let model_lifted: LiftedDiscretizedSystem<f64> = model.clone().try_into().unwrap();

    let rest_tlen = 0.02;
    let move_tlen = 0.05;
    let rest_samples = (rest_tlen / TS).round() as usize;
    let move_samples = (move_tlen / TS).round() as usize;
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

    let u = model_lifted.calculate_ptc_input_from_reference_state(r.clone());

    for i in 0..u.len() {
        storage.add(&[TS * i as f64, r[i][0], model.output[0], (r[i][0] - model.output[0])]).unwrap();
        model.update(&[u[i]]).unwrap();
    }

    storage.close().unwrap();
}
