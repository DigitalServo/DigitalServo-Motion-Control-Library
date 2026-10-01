use std::borrow::Borrow;

use nalgebra::{ComplexField, DMatrix, DVector, RealField};
use num_traits::Float;

use crate::{Continuous, Discrete, StateSpace, StateSpaceError, StateSpaceOrder, TransferFunction};

pub fn discretize_ssr<T: Float + ComplexField + RealField, S: Borrow<StateSpace<T, Continuous>>>(ssr: S, ts: T) -> Result<StateSpace<T, Discrete>, StateSpaceError> {
    let system = ssr.borrow();

    // Augmented matrix method
    let (a, b) = {
        let n = system.order.system;
        let m = system.order.input;

        let mut aug = DMatrix::<T>::zeros(n + m, n + m);
        aug.view_mut((0, 0), (n, n)).copy_from(&system.a);
        aug.view_mut((0, n), (n, m)).copy_from(&system.b);

        let aug_exp = aug.scale(ts).exp();

        let a = aug_exp.view((0, 0), (n, n)).into_owned();
        let b = aug_exp.view((0, n), (n, m)).into_owned();

        (a, b)
    };

    let c = system.c.clone();
    let d = system.d.clone();

    StateSpace::new(a, b, c, d)
}

#[derive(Clone)]
pub struct DiscretizedSystem<T> {
    /// Continuous-time model the system was discretized from.
    pub continuous: StateSpace<T, Continuous>,
    pub ssr: StateSpace<T, Discrete>,
    pub state: DVector<T>,
    pub output: DVector<T>,
    pub ts: T,
}

impl<T: Float + ComplexField + RealField> DiscretizedSystem<T> {
    pub fn from_ssr<S: Borrow<StateSpace<T, Continuous>>>(ssr: S, ts: T) -> Result<Self, StateSpaceError> {
        let continuous = ssr.borrow().clone();
        let ssr = discretize_ssr(&continuous, ts)?;
        let state = DVector::zeros(ssr.order.system);
        let output = DVector::zeros(ssr.order.output);

        Ok(Self { continuous, ssr, state, output, ts})
    }

    pub fn from_tf<S: Borrow<TransferFunction<T, Continuous>>>(tf_c: S, ts: T) -> Result<Self, StateSpaceError> {
        let tf_c = tf_c.borrow();

        let order = StateSpaceOrder {
            system: tf_c.denominator.len() - 1,
            input: 1,
            output: 1,
        };

        let scaler = T::one() / tf_c.denominator[0];
        let numerator: Vec<T> = tf_c.numerator.iter().map(|x| *x * scaler).collect();
        let denominator: Vec<T> = tf_c.denominator.iter().map(|x| *x * scaler).collect();

        // Coefficients of the denominator and numerator
        let ak: Vec<T> = denominator[1..].iter().rev().map(|x| -*x).collect();
        let bk: Vec<T> = numerator.into_iter().rev().collect();

        let dc_gain = bk.get(0).copied().ok_or(StateSpaceError::EmptySystem)?;

        let mut a = DMatrix::zeros(order.system, order.system);
        for i in 0..(order.system - 1) {
            a[(i, i + 1)] = T::one();
        }
        for (j, &val) in ak.iter().enumerate() {
            a[(order.system - 1, j)] = val;
        }

        let mut b = DMatrix::<T>::zeros(order.system, order.input);
        b[(order.system - 1, 0)] = dc_gain;

        let mut c = DMatrix::zeros(order.output, order.system);
        for i in 0..order.system {
            if let Some(&val) = bk.get(i) {
                c[(0, i)] = val / dc_gain;
            }
        }

        let d = DMatrix::zeros(order.output, order.input);

        let continuous = StateSpace::new(a, b, c, d)?;
        let ssr = discretize_ssr(&continuous, ts)?;
        let state = DVector::zeros(ssr.order.system);
        let output = DVector::zeros(ssr.order.output);

        Ok(Self { continuous, ssr, state, output, ts})
    }

    pub fn update(&mut self, u: &[T]) -> Result<Vec<T>, StateSpaceError> {
        if u.len() != self.ssr.order.input {
            return Err(StateSpaceError::InputVector {
                expected_row: self.ssr.order.input,
                actual_rows: u.len()
            })
        }

        let u: DVector<T> = DVector::from_row_slice(u);
        self.state = (&self.ssr.a * &self.state) + (&self.ssr.b * &u);
        self.output = (&self.ssr.c * &self.state) + (&self.ssr.d * &u);

        Ok(self.output.as_slice().to_vec())
    }
}

