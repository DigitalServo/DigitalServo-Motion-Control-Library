mod error;
pub use error::StateSpaceError;

use nalgebra::{ComplexField, DMatrix, RealField};
use num_traits::Float;
use std::marker::PhantomData;

use crate::{Continuous, Polynomial, TransferFunction};

/// Dimensions of a `StateSpace`.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct StateSpaceOrder {
    /// Number of states.
    pub system: usize,
    /// Number of inputs.
    pub input: usize,
    /// Number of outputs.
    pub output: usize,
}

/// `D` tells whether this is a continuous-time system (`dx/dt = Ax + Bu`, `Continuous`, the default)
/// or a discrete-time one (`x[k+1] = Ax[k] + Bu[k]`, `Discrete`).
#[derive(Clone, Debug)]
pub struct StateSpace<T, D = Continuous> {
    pub a: DMatrix<T>,
    pub b: DMatrix<T>,
    pub c: DMatrix<T>,
    pub d: DMatrix<T>,
    pub order: StateSpaceOrder,
    _domain: PhantomData<D>,
}

impl<T, D> StateSpace<T, D> {
    pub fn new(a: DMatrix<T>, b: DMatrix<T>, c: DMatrix<T>, d: DMatrix<T>) -> Result<Self, StateSpaceError> {

        let state_order = a.nrows();
        let input_order = b.ncols();
        let output_order = c.nrows();

        if a.ncols() != state_order {
            return Err(StateSpaceError::SystemMatrix {
                rows: a.nrows(),
                cols: a.ncols(),
            })
        }

        if b.nrows() != state_order {
            return Err(StateSpaceError::InputMatrix {
                expected_row: state_order,
                expected_col: input_order,
                actual_rows: b.nrows(),
                actual_cols: b.ncols(),
            })
        }

        if c.ncols() != state_order {
            return Err(StateSpaceError::OutputMatrix {
                expected_row: output_order,
                expected_col: state_order,
                actual_rows: c.nrows(),
                actual_cols: c.ncols(),
            })
        }

        if (d.nrows(), d.ncols()) != (output_order, input_order) {
            return Err(StateSpaceError::FeedthroughMatrix {
                expected_row: output_order,
                expected_col: input_order,
                actual_rows: d.nrows(),
                actual_cols: d.ncols(),
            })
        }

        let order = StateSpaceOrder {
            system: a.nrows(),
            input: b.ncols(),
            output: c.nrows(),
        };

        Ok(Self {a, b, c, d, order, _domain: PhantomData})
    }
}

impl<T: Float + ComplexField + RealField, D> StateSpace<T, D> {
    /// Transfer function `C (xI - A)^-1 B + D` of a single-input single-output system (`x` is `s`
    /// or `z` by the domain), by the Faddeev-LeVerrier algorithm:
    /// `adj(xI - A) = Σ_k M_k x^(n-1-k)`, `M_0 = I`, `M_k = A M_(k-1) + c_k I`,
    /// `det(xI - A) = x^n + c_1 x^(n-1) + ... + c_n` with `c_k = -tr(A M_(k-1)) / k`.
    pub fn transfer_function(&self) -> Result<TransferFunction<T, D>, StateSpaceError> {
        if self.order.input != 1 || self.order.output != 1 {
            return Err(StateSpaceError::NotSiso { inputs: self.order.input, outputs: self.order.output });
        }
        let n = self.order.system;
        let identity = DMatrix::<T>::identity(n, n);
        let mut m = identity.clone();
        let mut denominator = vec![T::one()];
        let mut adjugate = Vec::with_capacity(n);
        for k in 1..=n {
            adjugate.push((&self.c * &m * &self.b)[(0, 0)]);
            let am = &self.a * &m;
            let ck = -am.trace() / T::from(k).unwrap();
            denominator.push(ck);
            m = am + &identity * ck;
        }
        // C adj(xI - A) B + D det(xI - A): adjugate part has degree n - 1
        let d = self.d[(0, 0)];
        let numerator: Vec<T> = std::iter::once(T::zero())
            .chain(adjugate)
            .zip(&denominator)
            .map(|(cab, &det)| cab + d * det)
            .collect();
        Ok(TransferFunction::from_polynomials(Polynomial(numerator), Polynomial(denominator)))
    }
}

impl<T: Float + ComplexField + RealField> StateSpace<T, Continuous> {
    /// Controllable canonical realization of a proper `G(s) = N(s) / D(s)`:
    /// with `G = d + R(s) / D(s)` (`D` monic, `deg R < n`),
    /// `A` = companion matrix of `D`, `B = [0, ..., 0, 1]^T`, `C = [r_0, ..., r_(n-1)]`, `D = d`.
    /// A static gain (`n = 0`) has no state-space form (`EmptySystem`).
    pub fn controllable_canonical(tf: &TransferFunction<T, Continuous>) -> Result<Self, StateSpaceError> {
        let trim = |p: &Polynomial<T>| p.iter().copied().skip_while(|c| c.is_zero()).collect::<Vec<T>>();
        let numer = trim(&tf.numerator);
        let denom = trim(&tf.denominator);
        let n = denom.len().saturating_sub(1);
        if numer.len() > denom.len() {
            return Err(StateSpaceError::Improper { numerator: numer.len() - 1, denominator: n });
        }
        if n == 0 {
            return Err(StateSpaceError::EmptySystem);
        }

        // Monic denominator; direct term d and remainder R (ascending r_0 .. r_(n-1))
        let lead = denom[0];
        let denom: Vec<T> = denom.iter().map(|&c| c / lead).collect();
        let numer: Vec<T> = numer.iter().map(|&c| c / lead).collect();
        let (d, numer) = if numer.len() == denom.len() {
            // N = d D + R
            let d = numer[0];
            (d, numer.iter().zip(&denom).skip(1).map(|(&r, &a)| r - d * a).collect())
        } else {
            (T::zero(), numer)
        };
        let remainder: Vec<T> = numer.iter().rev().copied().collect();

        let mut a = DMatrix::<T>::zeros(n, n);
        for i in 0..n - 1 {
            a[(i, i + 1)] = T::one();
        }
        for j in 0..n {
            a[(n - 1, j)] = -denom[n - j];
        }
        let mut b = DMatrix::<T>::zeros(n, 1);
        b[(n - 1, 0)] = T::one();
        let mut c = DMatrix::<T>::zeros(1, n);
        for (j, &r) in remainder.iter().enumerate() {
            c[(0, j)] = r;
        }
        Self::new(a, b, c, DMatrix::from_element(1, 1, d))
    }

    /// Controllable canonical realization whose output equals a state at low frequency.
    /// With `N(s) = s^k N_1(s)`, `N_1(0) != 0` (`k` zeros at the origin), `N` and `D` are divided by
    /// `N_1(0)`: the state equation is `x = u / D(s)` and the output equation `y = N(s) x`, i.e.
    /// `x = [ξ, ξ', ..., ξ^(n-1)]` with `y ≈ ξ^(k) = x_(k+1)` at low frequency. In particular, for
    /// `N(0) != 0` (`k = 0`) `y = ξ` at DC, and `y = ξ` exactly without zeros.
    /// It is `controllable_canonical` with `B` scaled by `N'_1(0)` and `C` by `1 / N'_1(0)`
    /// (`N'` = numerator for the monic denominator).
    pub fn normalized_controllable_canonical(tf: &TransferFunction<T, Continuous>) -> Result<Self, StateSpaceError> {
        let mut ssr = Self::controllable_canonical(tf)?;
        // N'_1(0): lowest-order nonzero coefficient of the numerator (exact: trailing zeros of the
        // given coefficients), for the monic denominator.
        let lead = tf.denominator.iter().copied().find(|c| !c.is_zero()).unwrap_or_else(T::one);
        if let Some(&lowest) = tf.numerator.iter().rev().find(|c| !c.is_zero()) {
            let scale = lowest / lead;
            // x -> x / scale: B * scale, C / scale (D unchanged)
            ssr.b *= scale;
            ssr.c /= scale;
        }
        Ok(ssr)
    }
}
