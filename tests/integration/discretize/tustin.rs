//! Tustin (bilinear) transform: closed forms for transfer functions and state-space models.

use dsmc::discretize::{DiscretizeError, DiscretizeMethod, Tustin};
use dsmc::{tf, StateSpace, StateSpaceError, TransferFunction};
use nalgebra::{dmatrix, DMatrix};

use super::{assert_same_tf, TS};

/// `s = 2 (z - 1) / (ts (z + 1))` substituted into a first-order lag: with `a = 2 / ts`,
/// `g / (s + g) = g (z + 1) / ((a + g) z + (g - a))`.
#[test]
fn first_order() {
    let (g, a) = (10.0, 2.0 / TS);
    let gz = tf!("{g} / (s + {g})").discretize(Tustin, TS).unwrap();
    let expected = TransferFunction::discrete(&[g, g], &[a + g, g - a]);
    assert_same_tf(&gz, &expected, 1e-12, "Tustin");
}

/// Tustin keeps the state coordinates: `A_d = M (I + A ts/2)` with `M = (I - A ts/2)^-1`.
#[test]
fn state_space_formula() {
    let a = dmatrix![0.0, 1.0; -1000.0, -20.0];
    let ss = StateSpace::new(a.clone(), dmatrix![0.0; 1.0], dmatrix![1000.0, 0.0], dmatrix![0.0]).unwrap();
    let ss_z = ss.discretize(Tustin, TS).unwrap();
    let identity = DMatrix::<f64>::identity(2, 2);
    let m = (&identity - &a * (TS / 2.0)).try_inverse().unwrap();
    let expected_a = &m * (&identity + &a * (TS / 2.0));
    assert!((&ss_z.a - &expected_a).abs().max() < 1e-14, "{} vs {expected_a}", ss_z.a);
    assert_eq!(ss_z.c, dmatrix![1000.0, 0.0] * &m);

    // A has the eigenvalue 2/ts: I - A ts/2 is singular.
    let singular = StateSpace::new(dmatrix![2.0 / TS], dmatrix![1.0], dmatrix![1.0], dmatrix![0.0]).unwrap();
    assert_eq!(singular.discretize(Tustin, TS).unwrap_err(), StateSpaceError::SingularMatrix);
    assert_eq!(
        singular.discretize(DiscretizeMethod::Tustin, TS).unwrap_err(),
        DiscretizeError::StateSpace(StateSpaceError::SingularMatrix)
    );
}
