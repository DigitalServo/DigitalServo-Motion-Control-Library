//! Every discretization method on transfer functions and state-space models, by its own type and
//! by `DiscretizeMethod`.

use dsmc::discretize::{BackwardDifference, DiscretizeMethod, MatchedZ, MatchedZError, Tustin, Zoh, ZerosAtInfinity};
use dsmc::{Continuous, Discrete, DiscreteSystem, Polynomial, StateSpace, StateSpaceError, TransferFunction};
use nalgebra::dmatrix;

use super::{assert_same_tf, plants, step_response_ss, step_response_z, TS};

const METHODS: [DiscretizeMethod; 5] = [
    DiscretizeMethod::Zoh,
    DiscretizeMethod::Tustin,
    DiscretizeMethod::BackwardDifference,
    DiscretizeMethod::MatchedZ(ZerosAtInfinity::MinusOne),
    DiscretizeMethod::MatchedZ(ZerosAtInfinity::KeepOneDelay),
];

/// For every method, discretizing the transfer function and discretizing its realization give the
/// same discrete-time system, and the enum gives the same result as the method's own type.
#[test]
fn transfer_function_and_state_space_agree() {
    for g in plants() {
        let ss = StateSpace::try_from(&g).unwrap();
        for method in METHODS {
            let g_z = g.discretize(method, TS).unwrap();
            let ss_z = ss.discretize(method, TS).unwrap();
            let what = format!("{method:?} of {g:?}");
            assert_same_tf(&ss_z.transfer_function().unwrap(), &g_z, 1e-9, &what);

            let expected = step_response_z(&g_z, 200);
            let y = step_response_ss(ss_z, 200);
            for (k, (yk, e)) in y.iter().zip(&expected).enumerate() {
                assert!((yk - e).abs() < 1e-9 * e.abs().max(1.0), "{what} at k = {k}: {yk} vs {e}");
            }
        }

        assert_same_tf(&g.discretize(DiscretizeMethod::Zoh, TS).unwrap(), &g.discretize(Zoh, TS).unwrap(), 1e-9, "Zoh");
        assert_same_tf(&g.discretize(DiscretizeMethod::Tustin, TS).unwrap(), &g.discretize(Tustin, TS).unwrap(), 1e-9, "Tustin");
        assert_same_tf(&g.discretize(DiscretizeMethod::BackwardDifference, TS).unwrap(), &g.discretize(BackwardDifference, TS).unwrap(), 1e-9, "BackwardDifference");
        let option = ZerosAtInfinity::KeepOneDelay;
        assert_same_tf(&g.discretize(DiscretizeMethod::MatchedZ(option), TS).unwrap(), &g.discretize(MatchedZ(option), TS).unwrap(), 1e-9, "MatchedZ");
    }
}

/// Zoh, Tustin and BackwardDifference work for any number of inputs and outputs: each channel `(i, j)` is the
/// discretization of the continuous-time channel. The matched z-transform needs a single input
/// and output.
#[test]
fn multiple_inputs_and_outputs() {
    let ss = StateSpace::new(
        dmatrix![0.0, 1.0; -1000.0, -20.0],
        dmatrix![0.0, 1.0; 1.0, 0.5],
        dmatrix![1000.0, 0.0; 0.0, 1.0],
        dmatrix![0.0, 0.1; 0.0, 0.0],
    )
    .unwrap();
    let channel = |s: &StateSpace<f64, Continuous>, i: usize, j: usize| {
        StateSpace::<f64, Continuous>::new(s.a.clone(), s.b.columns(j, 1).into_owned(), s.c.rows(i, 1).into_owned(), s.d.view((i, j), (1, 1)).into_owned()).unwrap()
    };
    let channel_z = |s: &StateSpace<f64, Discrete>, i: usize, j: usize| {
        StateSpace::<f64, Discrete>::new(s.a.clone(), s.b.columns(j, 1).into_owned(), s.c.rows(i, 1).into_owned(), s.d.view((i, j), (1, 1)).into_owned()).unwrap()
    };
    for method in [DiscretizeMethod::Zoh, DiscretizeMethod::Tustin, DiscretizeMethod::BackwardDifference] {
        let ss_z = ss.discretize(method, TS).unwrap();
        for (i, j) in [(0, 0), (0, 1), (1, 0), (1, 1)] {
            let expected = channel(&ss, i, j).transfer_function().unwrap().discretize(method, TS).unwrap();
            assert_same_tf(&channel_z(&ss_z, i, j).transfer_function().unwrap(), &expected, 1e-9, &format!("{method:?} ({i}, {j})"));
        }
    }

    assert!(matches!(
        ss.discretize(MatchedZ(ZerosAtInfinity::MinusOne), TS),
        Err(MatchedZError::StateSpace(StateSpaceError::NotSiso { inputs: 2, outputs: 2 }))
    ));
}

/// A static gain is a state-space model without state and stays a gain with every method.
#[test]
fn static_gain() {
    let gain = TransferFunction::<f64, Continuous>::from_polynomials(Polynomial(vec![3.0]), Polynomial(vec![2.0]));
    let ss = StateSpace::try_from(&gain).unwrap();
    assert_eq!(ss.order.system, 0);
    assert_eq!(ss.d, dmatrix![1.5]);
    for method in METHODS {
        let mut from_ss = DiscreteSystem::from(ss.discretize(method, TS).unwrap());
        let mut from_tf = DiscreteSystem::try_from(&gain.discretize(method, TS).unwrap()).unwrap();
        assert_eq!(from_ss.update(&[2.0]).unwrap(), vec![3.0], "{method:?}");
        assert_eq!(from_tf.update(2.0), 3.0, "{method:?}");
    }
}
