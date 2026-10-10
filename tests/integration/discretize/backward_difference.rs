//! Backward difference (backward Euler): closed forms for transfer functions and state-space models.

use dsmc::discretize::{BackwardDifference, DiscretizeError, DiscretizeMethod};
use dsmc::{tf, StateSpace, StateSpaceError, TransferFunction};
use nalgebra::{dmatrix, DMatrix};

use super::{assert_same_tf, plants, TS};

/// `s = (z - 1) / (ts z)` substituted into a first-order lag:
/// `g / (s + g) = g ts z / ((1 + g ts) z - 1)`.
#[test]
fn first_order() {
    let g = 10.0;
    let gz = tf!("{g} / (s + {g})").discretize(BackwardDifference, TS).unwrap();
    let expected = TransferFunction::discrete(&[g * TS, 0.0], &[1.0 + g * TS, -1.0]);
    assert_same_tf(&gz, &expected, 1e-12, "BackwardDifference");
}

/// A second-order system with a zero: `(s + b) / (s^2 + a1 s + a0)` times `(ts z)^2` is
/// `ts z ((1 + b ts) z - 1) / ((1 + a1 ts + a0 ts^2) z^2 - (2 + a1 ts) z + 1)`.
#[test]
fn second_order() {
    let (b, a1, a0) = (50.0, 20.0, 1000.0);
    let gz = tf!("(s + {b}) / (s^2 + {a1}s + {a0})").discretize(BackwardDifference, TS).unwrap();
    let expected = TransferFunction::discrete(
        &[TS * (1.0 + b * TS), -TS, 0.0],
        &[1.0 + a1 * TS + a0 * TS * TS, -(2.0 + a1 * TS), 1.0],
    );
    assert_same_tf(&gz, &expected, 1e-12, "BackwardDifference");
}

/// Every pole `p` maps to `1 / (1 - p ts)`, and the DC gain `G(0) = G_d(1)` is kept.
#[test]
fn poles_and_dc_gain() {
    for g in plants() {
        let gz = g.discretize(BackwardDifference, TS).unwrap();
        let expected: Vec<_> = g.pz_map().poles.iter().map(|p| (1.0 - p * TS).inv()).collect();
        let actual = gz.pz_map().poles;
        assert_eq!(actual.len(), expected.len(), "{g:?}");
        for e in &expected {
            assert!(actual.iter().any(|a| (a - e).norm() < 1e-6), "{g:?}: {actual:?} vs {expected:?}");
        }

        let dc = g.denominator.last().unwrap();
        if *dc != 0.0 {
            let gain_s = g.numerator.last().unwrap() / dc;
            let gain_z = gz.numerator.iter().sum::<f64>() / gz.denominator.iter().sum::<f64>();
            assert!((gain_z - gain_s).abs() < 1e-9 * gain_s.abs().max(1.0), "{g:?}: {gain_z} vs {gain_s}");
        }
    }
}

/// `A_d = M`, `B_d = M B ts`, `C_d = C M`, `D_d = D + C M B ts` with `M = (I - A ts)^-1`.
#[test]
fn state_space_formula() {
    let a = dmatrix![0.0, 1.0; -1000.0, -20.0];
    let (b, c) = (dmatrix![0.0; 1.0], dmatrix![1000.0, 0.0]);
    let ss = StateSpace::new(a.clone(), b.clone(), c.clone(), dmatrix![0.0]).unwrap();
    let ss_z = ss.discretize(BackwardDifference, TS).unwrap();
    let m = (DMatrix::<f64>::identity(2, 2) - &a * TS).try_inverse().unwrap();
    assert_eq!(ss_z.a, m);
    assert_eq!(ss_z.c, &c * &m);
    assert!((&ss_z.b - &m * &b * TS).abs().max() < 1e-15);
    assert!((&ss_z.d - &c * &m * &b * TS).abs().max() < 1e-12);

    // A has the eigenvalue 1/ts: I - A ts is singular.
    let singular = StateSpace::new(dmatrix![1.0 / TS], dmatrix![1.0], dmatrix![1.0], dmatrix![0.0]).unwrap();
    assert_eq!(singular.discretize(BackwardDifference, TS).unwrap_err(), StateSpaceError::SingularMatrix);
    assert_eq!(
        singular.discretize(DiscretizeMethod::BackwardDifference, TS).unwrap_err(),
        DiscretizeError::StateSpace(StateSpaceError::SingularMatrix)
    );
}
