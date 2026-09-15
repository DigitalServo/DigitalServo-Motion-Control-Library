use std::{borrow::Borrow};

use nalgebra::{ComplexField, DMatrix, DVector, RealField};
use num_traits::Float;

use crate::{StateSpace, StateSpaceError, StateSpaceOrder, TransferFunction};

pub fn discretize_ssr<T: Float + ComplexField + RealField, S: Borrow<StateSpace<T>>>(ssr_c: S, ts: T) -> Result<StateSpace<T>, StateSpaceError> {
    let system = ssr_c.borrow();

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
    pub ssr: StateSpace<T>,
    pub state: DVector<T>,
    pub output: DVector<T>,
    pub ts: T,
}

impl<T: Float + ComplexField + RealField> DiscretizedSystem<T> {
    pub fn from_ssr<S: Borrow<StateSpace<T>>>(ssr_c: S, ts: T) -> Result<Self, StateSpaceError> {
        let ssr = discretize_ssr(ssr_c, ts)?;
        let state = DVector::zeros(ssr.order.system);
        let output = DVector::zeros(ssr.order.output);

        Ok(Self { ssr, state, output, ts})
    }

    pub fn from_tf<S: Borrow<TransferFunction<T>>>(tf_c: S, ts: T) -> Result<Self, StateSpaceError> {
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

        let ssr_c = StateSpace::new(a, b, c, d)?;
        let ssr = discretize_ssr(ssr_c, ts)?;
        let state = DVector::zeros(ssr.order.system);
        let output = DVector::zeros(ssr.order.output);

        Ok(Self { ssr, state, output, ts})
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


pub struct LiftedDiscretizedSystem<T> {
    pub ssr: StateSpace<T>,
    pub ts: T,
    pub order: u32,
    inv_b: DMatrix<T>,
}

impl<T: Float + ComplexField + RealField> TryInto<LiftedDiscretizedSystem<T>> for DiscretizedSystem<T> {
    type Error = StateSpaceError;

    fn try_into(self) -> Result<LiftedDiscretizedSystem<T>, Self::Error> {
        let system = self.borrow();

        let n = system.ssr.order.system;
        let m = system.ssr.order.input;

        let order = n as u32;

        let a = &system.ssr.a;
        let b = &system.ssr.b;

        let a_lifted = a.pow(order);

        // b_lifted = [a^(order-1)*b, a^(order-2)*b, ..., a*b, b]
        let mut b_lifted = DMatrix::<T>::zeros(n, m * order as usize);
        let mut a_power = DMatrix::<T>::identity(n, n);
        for k in 0..order as usize {
            let col = (order as usize - 1 - k) * m;
            b_lifted.columns_mut(col, m).copy_from(&(&a_power * b));
            a_power = a * &a_power;
        }

        let d = DMatrix::<T>::zeros(system.ssr.order.output, order as usize);

        let inv_b = b_lifted.clone()
            .try_inverse()
            .ok_or(StateSpaceError::SingularMatrix)?;

        let ssr = StateSpace::new(a_lifted.clone(), b_lifted.clone(), system.ssr.c.clone(), d)?;

        Ok(LiftedDiscretizedSystem {
            ssr,
            ts: system.ts,
            order,
            inv_b
        })
    }
}


impl<T: Float + ComplexField + RealField> LiftedDiscretizedSystem<T> {
    pub fn calculate_ptc_input(&self, r: Vec<Vec<T>>) -> Vec<T> {

        let mut u = Vec::<T>::with_capacity(r.len());

        let series: Vec<Vec<T>> = r
            .chunks(self.order as usize)
            .map(|x| x[0].clone())
            .collect();

        for points in series.windows(2) {
            let r_start = DVector::from(points[0].clone());
            let r_end = DVector::from(points[1].clone());

            let u_effort = r_end - &self.ssr.a * r_start;
            let u_local = &self.inv_b * u_effort.clone();

            u.extend(u_local.iter());
        }

        u
    }
}
