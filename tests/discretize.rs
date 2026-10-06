//! Every discretization method on transfer functions and state-space models, by its own type and
//! by `DiscretizeMethod`.

use dsmc::discretize::{DiscretizeError, DiscretizeMethod, MatchedZ, MatchedZError, Tustin, Zoh, ZerosAtInfinity};
use dsmc::{tf, Continuous, Discrete, DiscreteSystem, Polynomial, StateSpace, StateSpaceError, TransferFunction};
use nalgebra::{dmatrix, DMatrix};

const TS: f64 = 1e-3;

const METHODS: [DiscretizeMethod; 4] = [
    DiscretizeMethod::Zoh,
    DiscretizeMethod::Tustin,
    DiscretizeMethod::MatchedZ(ZerosAtInfinity::MinusOne),
    DiscretizeMethod::MatchedZ(ZerosAtInfinity::KeepOneDelay),
];

fn plants() -> Vec<TransferFunction<f64, Continuous>> {
    vec![
        tf!("100 / (s + 100)"),
        tf!("1000 / (s^2 + 20s + 1000)"),
        tf!("(1 - 0.01s) / ((s + 20)^2)"),
        tf!("(s + 50) / ((s + 10) (s^2 + 40s + 10000))"),
        tf!("(2s + 30) / (s + 10)"),
        tf!("1 / (s (0.02s + 1))"),
    ]
}

/// Coefficients with leading zeros removed, divided by the leading denominator coefficient.
fn normalized(g: &TransferFunction<f64, Discrete>) -> (Vec<f64>, Vec<f64>) {
    let trim = |p: &Polynomial<f64>| p.iter().copied().skip_while(|c| c.abs() < 1e-14).collect::<Vec<f64>>();
    let (n, d) = (trim(&g.numerator), trim(&g.denominator));
    (n.iter().map(|c| c / d[0]).collect(), d.iter().map(|c| c / d[0]).collect())
}

fn assert_same_tf(a: &TransferFunction<f64, Discrete>, b: &TransferFunction<f64, Discrete>, what: &str) {
    let (an, ad) = normalized(a);
    let (bn, bd) = normalized(b);
    assert_eq!((an.len(), ad.len()), (bn.len(), bd.len()), "{what}: {a:?} vs {b:?}");
    for (x, y) in an.iter().chain(&ad).zip(bn.iter().chain(&bd)) {
        assert!((x - y).abs() < 1e-9 * y.abs().max(1.0), "{what}: {a:?} vs {b:?}");
    }
}

fn step_response(system: &mut DiscreteSystem<f64>, samples: usize) -> Vec<f64> {
    (0..samples).map(|_| system.update(&[1.0]).unwrap()[0]).collect()
}

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
            assert_same_tf(&ss_z.transfer_function().unwrap(), &g_z, &what);

            let mut siso = DiscreteSystem::try_from(&g_z).unwrap();
            let expected: Vec<f64> = (0..200).map(|_| siso.update(1.0)).collect();
            let y = step_response(&mut DiscreteSystem::from(ss_z), 200);
            for (k, (yk, e)) in y.iter().zip(&expected).enumerate() {
                assert!((yk - e).abs() < 1e-9 * e.abs().max(1.0), "{what} at k = {k}: {yk} vs {e}");
            }
        }

        assert_same_tf(&g.discretize(DiscretizeMethod::Zoh, TS).unwrap(), &g.discretize(Zoh, TS).unwrap(), "Zoh");
        assert_same_tf(&g.discretize(DiscretizeMethod::Tustin, TS).unwrap(), &g.discretize(Tustin, TS).unwrap(), "Tustin");
        let option = ZerosAtInfinity::KeepOneDelay;
        assert_same_tf(&g.discretize(DiscretizeMethod::MatchedZ(option), TS).unwrap(), &g.discretize(MatchedZ(option), TS).unwrap(), "MatchedZ");
    }
}

/// Tustin keeps the state coordinates: `A_d = M (I + A ts/2)` with `M = (I - A ts/2)^-1`.
#[test]
fn tustin_state_space_formula() {
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

/// Zoh and Tustin work for any number of inputs and outputs: each channel `(i, j)` is the
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
    for method in [DiscretizeMethod::Zoh, DiscretizeMethod::Tustin] {
        let ss_z = ss.discretize(method, TS).unwrap();
        for (i, j) in [(0, 0), (0, 1), (1, 0), (1, 1)] {
            let expected = channel(&ss, i, j).transfer_function().unwrap().discretize(method, TS).unwrap();
            assert_same_tf(&channel_z(&ss_z, i, j).transfer_function().unwrap(), &expected, &format!("{method:?} ({i}, {j})"));
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
