use dsmc::discretize::exact_discretize::DiscretizedSystem;
use dsmc::feedforward::ptc::{LiftedDiscretizedSystem, PtcError, ReferenceSignal};
use dsmc::trajectory;
use dsmc::laplace_transform::{PiecewisePolynomial, StableInverseError};
use dsmc::{tf, StateSpace, TransferFunction};
use nalgebra::DMatrix;

const TS: f64 = 1e-4;
const REST: f64 = 0.02;
const DURATION: f64 = 0.05;

/// Relative degree 4 with a zero at +1000 rad/s.
fn plant() -> TransferFunction<f64> {
    tf!("(1 - 0.001s) / (s (0.005s + 1) (0.001s + 1) (0.0005s + 1) (0.0002s + 1))")
}

fn y_d() -> PiecewisePolynomial<f64> {
    trajectory::smoothstep::piecewise(1.0, DURATION, REST, 4)
}

/// Simulate `u` on `model` and return the largest |y - y_d| at frame instants.
fn max_frame_error(mut model: DiscretizedSystem<f64>, u: &[f64], order: usize) -> f64 {
    let y = y_d();
    let mut max_error: f64 = 0.0;
    for (i, &ui) in u.iter().enumerate() {
        // y[i] at t = i ts; frames start at i = 0, n, 2n, ...
        let yi = model.update(&[ui]).unwrap()[0];
        if i % order == 0 {
            max_error = max_error.max((yi - y.value(i as f64 * TS)).abs());
        }
    }
    max_error
}

fn samples() -> usize {
    ((2.0 * REST + DURATION) / TS).round() as usize
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
    let u = lifted.calculate_ptc_input_for_reference_output(&y_d(), samples()).unwrap();
    let error = max_frame_error(model, &u, 5);
    assert!(error < 1e-9, "max tracking error at frames: {error:e}");

    // The state reference is in the cascade's coordinates: x1 = ξ (position), x2 = ξ'.
    let x_d = lifted.state_reference_from_output(&y_d()).unwrap();
    let canonical = y_d().to_state_reference(&plant()).unwrap();
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
        lifted.calculate_ptc_input_for_reference_output(&y_d(), 100),
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
        lifted.calculate_ptc_input_for_reference_output(&y_d(), 100).unwrap_err(),
        PtcError::Feedthrough
    );
}
