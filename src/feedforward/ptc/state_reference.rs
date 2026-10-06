//! State reference for perfect tracking control by stable inversion.

use crate::system::principal_part;
use crate::laplace_transform::{jump_rational, DelayedRationalSum, PiecewisePolynomial, StableInverse, StableInverseError};
use crate::{vieta_formula, Continuous, PartialFraction, PoleTerm, Polynomial, TransferFunction};
use num_complex::Complex;
use num_traits::{Float, Zero};
use std::ops::AddAssign;

/// State reference `x_d(t) = [ξ_d, ξ_d', ..., ξ_d^(n-1)]` for perfect tracking control by stable inversion.
///
/// For `G(s) = N(s) / D(s)` (`n = deg D`) in the controllable canonical realization normalized by
/// `N(0)` (`StateSpace::normalized_controllable_canonical`, used by `LiftedDiscretizedSystem::new`),
/// the state is `x = [ξ, ξ', ..., ξ^(n-1)]` with `y = N(s) / N(0) ξ`, hence for a desired output `y_d`
///
/// ```text
/// ξ_d^(j)(t) = L^-1[ F_j(s) Y_d(s) ](t),    F_j(s) = N(0) s^j / N(s)    (bilateral, j = 0..n-1)
/// ```
///
/// Only `N(s)` is inverted: its stable zeros give causal terms, its unstable zeros anti-causal ones
/// (pre-actuation before the trajectory starts). Everything is evaluated in closed form (no convolution).
#[derive(Clone, Debug)]
pub struct StateReference<T> {
    /// `states[j]` = `[(τ_i, bilateral inverse of F_j(s) R_i(s))]` for `Y_d = Σ e^(-s τ_i) R_i(s)`.
    states: Vec<Vec<(T, StableInverse<T>)>>,
    /// For piecewise-polynomial references, the part of `ξ_d^(j)` from the poles of `Y_d` at
    /// `s = 0` is `Σ_k f_{j,k} y_d^(k)(t)` (`f_{j,k}`: Taylor coefficients of `F_j` at 0),
    /// evaluated from the local piece; `states` then holds only the zeros-of-N terms.
    polynomial: Option<(PiecewisePolynomial<T>, Vec<Vec<T>>)>,
}

/// Desired outputs `y_d` that can be turned into a state reference: `DelayedRationalSum` (any delayed
/// rationals) and `PiecewisePolynomial` (evaluated without cancellation long after the move).
pub trait ReferenceSignal<T> {
    /// State reference for perfect tracking control that makes the output of `plant` follow this
    /// signal (see [`StateReference`]), e.g. `y_d.to_state_reference(&plant)`.
    /// Requires that `N(s)` has no zero on the imaginary axis and that this signal is smooth enough
    /// that no state reference contains impulses.
    fn to_state_reference(&self, plant: &TransferFunction<T, Continuous>) -> Result<StateReference<T>, StableInverseError>;
}

/// Numerator of the plant with its zeros (as poles of `1 / N(s)`), checked to be off the imaginary axis.
struct Numerator<T> {
    /// Descending, leading zeros removed.
    coefficients: Vec<T>,
    /// `N(0)` (`y = N(s) / N(0) ξ`)
    scale: T,
    zeros: Vec<PoleTerm<T>>,
    /// `n = deg D`
    order: usize,
}

impl<T: Float + AddAssign> Numerator<T> {
    fn of(plant: &TransferFunction<T, Continuous>) -> Result<Self, StableInverseError> {
        let coefficients: Vec<T> = plant.numerator.iter().copied().skip_while(|c| c.is_zero()).collect();
        let order = plant.denominator.iter().skip_while(|c| c.is_zero()).count().saturating_sub(1);
        let Some(&scale) = coefficients.last() else {
            return Err(StableInverseError::ZeroSystem);
        };
        let zeros = TransferFunction::<T, Continuous>::from_polynomials(
            Polynomial(vec![T::one()]),
            Polynomial(coefficients.clone()),
        )
        .partial_fraction()
        .terms;
        if let Some(z) = zeros.iter().find(|z| z.pole.re.is_zero()) {
            return Err(StableInverseError::PoleOnImaginaryAxis { re: to_f64(z.pole.re), im: to_f64(z.pole.im) });
        }
        Ok(Self { coefficients, scale, zeros, order })
    }

    fn degree(&self) -> usize {
        self.coefficients.len() - 1
    }

    /// `N(0) s^j` (descending)
    fn scaled_s_j(&self, j: usize) -> Polynomial<T> {
        let mut p = Polynomial(vec![T::zero(); j + 1]);
        p[0] = self.scale;
        p
    }
}

impl<T: Float + AddAssign> ReferenceSignal<T> for DelayedRationalSum<T> {
    /// Each `F_j(s) R_i(s)` is expanded into partial fractions as a whole. Poles of `R_i` and stable
    /// zeros of `N` are causal, unstable zeros of `N` anti-causal.
    fn to_state_reference(&self, plant: &TransferFunction<T, Continuous>) -> Result<StateReference<T>, StableInverseError> {
        let numer = Numerator::of(plant)?;

        // Pick Re s = sigma between the causal poles and the anti-causal ones.
        let causal_re = self
            .terms
            .iter()
            .flat_map(|c| c.rational.partial_fraction().terms)
            .map(|term| term.pole.re)
            .chain(numer.zeros.iter().map(|z| z.pole.re).filter(|&re| re < T::zero()))
            .reduce(T::max);
        let anticausal_re = numer.zeros.iter().map(|z| z.pole.re).filter(|&re| re > T::zero()).reduce(T::min);
        let sigma = match (causal_re, anticausal_re) {
            (Some(lo), Some(hi)) if lo < hi => (lo + hi) / T::from(2.0).unwrap(),
            (Some(lo), Some(hi)) => {
                return Err(StableInverseError::NoRegionOfConvergence { causal_re: to_f64(lo), anticausal_re: to_f64(hi) });
            }
            (Some(lo), None) => lo + T::one(),
            (None, Some(hi)) => hi - T::one(),
            (None, None) => T::zero(),
        };

        let n_poly = Polynomial(numer.coefficients.clone());
        let states = (0..numer.order)
            .map(|j| {
                let scaled_s_j = numer.scaled_s_j(j);
                self.terms
                    .iter()
                    .map(|c| {
                        let pf = TransferFunction::<T, Continuous>::from_polynomials(
                            &scaled_s_j * &c.rational.numerator,
                            &n_poly * &c.rational.denominator,
                        )
                        .partial_fraction();
                        if !pf.direct.is_empty() {
                            return Err(StableInverseError::NotSmoothEnough { state: j });
                        }
                        Ok((c.delay, StableInverse::bilateral_with_abscissa(pf, sigma)?))
                    })
                    .collect::<Result<Vec<_>, _>>()
            })
            .collect::<Result<Vec<_>, _>>()?;

        Ok(StateReference { states, polynomial: None })
    }
}

impl<T: Float + AddAssign> ReferenceSignal<T> for PiecewisePolynomial<T> {
    /// Split by the poles of `F_j(s) R_i(s)`: those at `s = 0` (from the jumps) sum up to
    /// `F_j(d/dt) y_d` on the local piece, and only the zeros of `N` need partial fractions.
    fn to_state_reference(&self, plant: &TransferFunction<T, Continuous>) -> Result<StateReference<T>, StableInverseError> {
        let numer = Numerator::of(plant)?;
        let jumps = self.jumps();

        // F_j R_i is proper iff j - deg N <= (order of the jump), i.e. y_d^(j - deg N) has no impulse.
        if let Some(lowest) = jumps.iter().map(|jump| jump.order()).min() {
            let first_bad = numer.degree() + lowest + 1;
            if first_bad < numer.order {
                return Err(StableInverseError::NotSmoothEnough { state: first_bad });
            }
        }

        // f_{j,k}: Taylor coefficients of N(0) s^j / N(s) at s = 0, by power-series division.
        let count = self.degree() + 1;
        let n_ascending: Vec<T> = numer.coefficients.iter().rev().copied().collect();
        let series: Vec<Vec<T>> = (0..numer.order)
            .map(|j| {
                let mut f = vec![T::zero(); count];
                for k in 0..count {
                    let mut acc = if k == j { numer.scale } else { T::zero() };
                    for l in 1..=k.min(numer.degree()) {
                        acc = acc - n_ascending[l] * f[k - l];
                    }
                    f[k] = acc / n_ascending[0];
                }
                f
            })
            .collect();

        let lead = Complex::from(numer.coefficients[0]);
        let states = (0..numer.order)
            .map(|j| {
                jumps
                    .iter()
                    .map(|jump| {
                        // F_j R_i = N(0) s^j r(s) / (N(s) s^M)
                        let rational = jump_rational(jump);
                        let mut num: Vec<Complex<T>> =
                            rational.numerator.iter().map(|&c| Complex::from(c * numer.scale)).collect();
                        num.extend(std::iter::repeat_n(Complex::zero(), j));
                        let power_of_s = rational.denominator.len() - 1;

                        let terms = numer
                            .zeros
                            .iter()
                            .enumerate()
                            .map(|(i, z)| {
                                let others: Vec<Complex<T>> = numer
                                    .zeros
                                    .iter()
                                    .enumerate()
                                    .filter(|&(k, _)| k != i)
                                    .flat_map(|(_, o)| std::iter::repeat_n(o.pole, o.multiplicity()))
                                    .collect();
                                // (N(s) / (s - z)^m) s^M
                                let mut rest: Vec<Complex<T>> =
                                    vieta_formula(&others).0.iter().map(|&c| c * lead).collect();
                                rest.extend(std::iter::repeat_n(Complex::zero(), power_of_s));
                                PoleTerm { pole: z.pole, residues: principal_part(&num, &rest, z.pole, z.multiplicity()) }
                            })
                            .collect();
                        let pf = PartialFraction { direct: Polynomial(vec![]), terms };
                        Ok((jump.time, StableInverse::bilateral(pf)?))
                    })
                    .collect::<Result<Vec<_>, _>>()
            })
            .collect::<Result<Vec<_>, _>>()?;

        Ok(StateReference { states, polynomial: Some((self.clone(), series)) })
    }
}

impl<T: Float> StateReference<T> {
    /// Number of states `n`.
    pub fn order(&self) -> usize {
        self.states.len()
    }

    /// `x_d(t) = [ξ_d(t), ξ_d'(t), ..., ξ_d^(n-1)(t)]`.
    pub fn state(&self, t: T) -> Vec<T> {
        let mut x: Vec<T> = self
            .states
            .iter()
            .map(|components| {
                components
                    .iter()
                    .fold(T::zero(), |acc, (delay, inv)| acc + inv.impulse_response(t - *delay))
            })
            .collect();
        if let Some((y_d, series)) = &self.polynomial {
            let count = series.first().map_or(0, |f| f.len());
            let derivatives = y_d.derivatives(t, count);
            for (xj, f) in x.iter_mut().zip(series) {
                *xj = *xj + f.iter().zip(&derivatives).fold(T::zero(), |acc, (&fk, &dk)| acc + fk * dk);
            }
        }
        x
    }

    /// `x_d(t0 + k ts)` for k = 0..samples, e.g. as the reference of `LiftedDiscretizedSystem::calculate_ptc_input_for_reference_state`.
    pub fn sample(&self, t0: T, ts: T, samples: usize) -> Vec<Vec<T>> {
        (0..samples).map(|k| self.state(t0 + ts * T::from(k).unwrap())).collect()
    }
}

fn to_f64<T: Float>(x: T) -> f64 {
    x.to_f64().unwrap_or(f64::NAN)
}
