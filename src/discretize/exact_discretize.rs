use std::borrow::Borrow;

use nalgebra::{ComplexField, DMatrix, DVector, RealField};
use num_traits::Float;
use thiserror::Error;

use crate::{
    Continuous, Discrete, Polynomial, ReferenceSignal, StableInverseError, StateSpace, StateSpaceError,
    StateSpaceOrder, TransferFunction,
};

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


pub struct LiftedDiscretizedSystem<T> {
    /// Continuous-time model the system was discretized from.
    pub continuous: StateSpace<T, Continuous>,
    pub ssr: StateSpace<T, Discrete>,
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
            continuous: system.continuous.clone(),
            ssr,
            ts: system.ts,
            order,
            inv_b
        })
    }
}


#[derive(Clone, Debug, Error, PartialEq)]
pub enum PtcError {
    #[error("Perfect tracking from an output reference supports SISO systems only, got {inputs} inputs and {outputs} outputs")]
    NotSiso { inputs: usize, outputs: usize },

    #[error("The system has a direct feedthrough term (D != 0)")]
    Feedthrough,

    #[error("(A, B) is not controllable")]
    Uncontrollable,

    #[error(transparent)]
    StableInverse(#[from] StableInverseError),
}

impl<T: Float + ComplexField + RealField> LiftedDiscretizedSystem<T> {
    /// Perfect tracking control input from a state reference given at every sample:
    /// `u[k] = B_lifted^-1 (r[(k+1)n] - A^n r[kn])` for each frame of `n` samples
    /// (only every `n`-th element of `r` is used).
    pub fn calculate_ptc_input_from_reference_state(&self, r: Vec<Vec<T>>) -> Vec<T> {
        let frames: Vec<DVector<T>> = r
            .chunks(self.order as usize)
            .map(|x| DVector::from(x[0].clone()))
            .collect();
        self.inputs_from_frame_states(&frames)
    }

    /// `B_lifted^-1 (x[k+1] - A^n x[k])` for consecutive frame states.
    fn inputs_from_frame_states(&self, frames: &[DVector<T>]) -> Vec<T> {
        let mut u = Vec::<T>::with_capacity(frames.len() * self.order as usize);
        for points in frames.windows(2) {
            let u_effort = &points[1] - &self.ssr.a * &points[0];
            let u_local = &self.inv_b * u_effort;
            u.extend(u_local.iter());
        }
        u
    }

    /// Perfect tracking control input that makes the output follow `y_d`, for `samples` samples
    /// starting at `t0` (rounded up to whole frames of `n` samples).
    ///
    /// The state reference is obtained by stable inversion (bilateral Laplace transform) of the
    /// continuous-time plant `G(s) = C (sI - A)^-1 B`, see `TransferFunction::state_reference`.
    /// Unstable zeros of `G` make it non-causal: the input starts before the trajectory does
    /// (pre-actuation), so leave enough rest time before the move.
    pub fn calculate_ptc_input_from_reference_output<S: ReferenceSignal<T>>(
        &self,
        y_d: &S,
        t0: T,
        samples: usize,
    ) -> Result<Vec<T>, PtcError> {
        let reference = self.reference_state(y_d)?;
        let n = self.order as usize;
        let frame_time = self.ts * T::from(n).unwrap();
        let frames: Vec<DVector<T>> = (0..=samples.div_ceil(n))
            .map(|k| reference(t0 + frame_time * T::from(k).unwrap()))
            .collect();
        Ok(self.inputs_from_frame_states(&frames))
    }

    /// State reference `x_d(t)` (in this system's state coordinates) that makes the output follow `y_d`.
    pub fn reference_state<S: ReferenceSignal<T>>(&self, y_d: &S) -> Result<impl Fn(T) -> DVector<T> + use<T, S>, PtcError> {
        let sys = &self.continuous;
        if sys.order.input != 1 || sys.order.output != 1 {
            return Err(PtcError::NotSiso { inputs: sys.order.input, outputs: sys.order.output });
        }
        if sys.d.iter().any(|d| !d.is_zero()) {
            return Err(PtcError::Feedthrough);
        }

        let plant = transfer_function(&sys.a, &sys.b, &sys.c);
        let reference = plant.state_reference(y_d)?;

        // x = T x_c with the controllable canonical realization (A_c, B_c) of `DiscretizedSystem::from_tf`
        // (in which `StateReference` is expressed): T = W(A, B) W(A_c, B_c)^-1.
        let canonical = DiscretizedSystem::from_tf(&plant, T::one()).map_err(|_| PtcError::Uncontrollable)?.continuous;
        let w = controllability_matrix(&sys.a, &sys.b);
        let w_c = controllability_matrix(&canonical.a, &canonical.b);
        let w_c_inv = w_c.try_inverse().ok_or(PtcError::Uncontrollable)?;
        let transform = w * w_c_inv;
        if transform.clone().try_inverse().is_none() {
            return Err(PtcError::Uncontrollable);
        }

        Ok(move |t: T| &transform * DVector::from(reference.state(t)))
    }
}

/// `[B, AB, ..., A^(n-1) B]` (single input).
fn controllability_matrix<T: Float + ComplexField + RealField>(a: &DMatrix<T>, b: &DMatrix<T>) -> DMatrix<T> {
    let n = a.nrows();
    let mut w = DMatrix::<T>::zeros(n, n);
    let mut column = b.column(0).into_owned();
    for k in 0..n {
        w.set_column(k, &column);
        column = a * column;
    }
    w
}

/// `C (sI - A)^-1 B` (single input / output) by the Faddeev-LeVerrier algorithm:
/// `adj(sI - A) = Σ_k M_k s^(n-1-k)`, `M_0 = I`, `M_k = A M_(k-1) + c_k I`,
/// `det(sI - A) = s^n + c_1 s^(n-1) + ... + c_n` with `c_k = -tr(A M_(k-1)) / k`.
fn transfer_function<T: Float + ComplexField + RealField>(
    a: &DMatrix<T>,
    b: &DMatrix<T>,
    c: &DMatrix<T>,
) -> TransferFunction<T, Continuous> {
    let n = a.nrows();
    let identity = DMatrix::<T>::identity(n, n);
    let mut m = identity.clone();
    let mut denominator = vec![T::one()];
    let mut numerator = Vec::with_capacity(n);
    for k in 1..=n {
        numerator.push((c * &m * b)[(0, 0)]);
        let am = a * &m;
        let ck = -am.trace() / T::from(k).unwrap();
        denominator.push(ck);
        m = am + &identity * ck;
    }
    TransferFunction::from_polynomials(Polynomial(numerator), Polynomial(denominator))
}
