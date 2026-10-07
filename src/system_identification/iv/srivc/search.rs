//! Search of the model structure `(n, m, q, nk)` by identification and validation.

use std::fmt;
use std::ops::{AddAssign, MulAssign};

use nalgebra::{ComplexField, RealField};
use num_traits::Float;
use rustfft::FftNum;

use super::{identify, identify_with_prefilter, Initialization, Prefilter, SrivcError, SrivcOptions, SrivcResult};
use crate::system_identification::preprocessing::high_pass;
use crate::system_identification::validation::{Check, CheckResult, Report, Validation, ValidationError};

/// Model structure `e^(-nk ts s) B(s) / (s^q A(s))` of SRIVC: `deg A = n = denominator_order`,
/// `deg B = m = numerator_order`, `q = integrators` (poles fixed at the origin, see
/// `identify_with_prefilter`), `nk = input_delay` \[samples\], always in this order `(n, m, q, nk)`
/// (constructor, `grid`, display).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Structure {
    /// `n`.
    pub denominator_order: usize,
    /// `m`.
    pub numerator_order: usize,
    /// `q`.
    pub integrators: usize,
    /// `nk` \[samples\].
    pub input_delay: usize,
}

impl Structure {
    /// `(n, m, q, nk)`: the denominator order first, e.g. `b_0 / (s (s^2 + a_1 s + a_2))` with a
    /// delay of 3 samples is `Structure::new(2, 0, 1, 3)`.
    pub fn new(denominator_order: usize, numerator_order: usize, integrators: usize, input_delay: usize) -> Self {
        Self { denominator_order, numerator_order, integrators, input_delay }
    }

    /// Number of estimated parameters of `B / A`, `n + m + 1` (the penalty of BIC / AIC; the poles
    /// at the origin are fixed, not estimated).
    pub fn parameters(&self) -> usize {
        self.denominator_order + self.numerator_order + 1
    }

    /// Every structure with `n` in `denominator_orders`, `m` in `numerator_orders` (`m <= n`), `q`
    /// in `integrators` and `nk` in `input_delays` (arguments in the order `(n, m, q, nk)`).
    pub fn grid(
        denominator_orders: impl IntoIterator<Item = usize>,
        numerator_orders: impl IntoIterator<Item = usize> + Clone,
        integrators: impl IntoIterator<Item = usize> + Clone,
        input_delays: impl IntoIterator<Item = usize> + Clone,
    ) -> Vec<Self> {
        let mut structures = Vec::new();
        for n in denominator_orders {
            for m in numerator_orders.clone().into_iter().filter(|&m| m <= n) {
                for q in integrators.clone() {
                    for nk in input_delays.clone() {
                        structures.push(Self::new(n, m, q, nk));
                    }
                }
            }
        }
        structures
    }
}

impl fmt::Display for Structure {
    /// `(n, m, q, nk) = (4, 2, 1, 8)`.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "(n, m, q, nk) = ({}, {}, {}, {})", self.denominator_order, self.numerator_order, self.integrators, self.input_delay)
    }
}

/// Options of `search`.
#[derive(Clone, Debug)]
pub struct SearchOptions<T> {
    /// Starting point of the SRIVC iterations of every candidate.
    pub initialization: Initialization<T>,
    /// SRIVC options (`input_delay` is taken from each structure; their `evaluated_from` applies to
    /// the identification data, `SearchOptions::evaluated_from` to the validation data).
    pub srivc: SrivcOptions<T>,
    /// Tests run on the validation data; the information criteria (with `Structure::parameters`)
    /// are added before them.
    pub checks: Vec<Check<T>>,
    /// Confidence level of the tests.
    pub confidence: T,
    /// First evaluated sample of the validation data \[samples\] (`Validation::evaluated_from` takes
    /// seconds), e.g. one period of a periodic input to leave out the transient from rest (required
    /// by `Check::Lines`).
    pub evaluated_from: usize,
    /// Cutoff `ω_c` \[rad/s\] of the prefilters, needed by the structures with `q > 0`: each
    /// candidate is identified with `Prefilter::new(q, ω_c)`, and the validation data are
    /// high-passed alike for all of them (`preprocessing::high_pass` at `ω_c`, of order
    /// `q_max + 1` over the structures), so that the models with poles at the origin do not drift
    /// and the BIC are compared on the same data. `None`: no prefilter, `q = 0` only.
    pub prefilter: Option<T>,
}

impl<T: Float> SearchOptions<T> {
    /// Default SRIVC options, evaluation from the first sample, no prefilter.
    pub fn new(initialization: Initialization<T>, checks: Vec<Check<T>>, confidence: T) -> Self {
        Self { initialization, srivc: SrivcOptions::default(), checks, confidence, evaluated_from: 0, prefilter: None }
    }
}

/// What happened to one candidate structure.
#[derive(Clone, Debug)]
pub enum Outcome<T> {
    /// SRIVC failed (e.g. singular normal equations).
    NotIdentified(SrivcError),
    /// The identified model diverges on the validation input.
    Unstable(SrivcResult<T>),
    /// Identified and validated.
    Validated { result: SrivcResult<T>, report: Report<T> },
}

/// One candidate of `search`.
#[derive(Clone, Debug)]
pub struct Candidate<T> {
    pub structure: Structure,
    pub outcome: Outcome<T>,
}

impl<T: Copy> Candidate<T> {
    /// The identification result, unless SRIVC failed.
    pub fn result(&self) -> Option<&SrivcResult<T>> {
        match &self.outcome {
            Outcome::NotIdentified(_) => None,
            Outcome::Unstable(result) | Outcome::Validated { result, .. } => Some(result),
        }
    }

    /// The validation report, if validated.
    pub fn report(&self) -> Option<&Report<T>> {
        match &self.outcome {
            Outcome::Validated { report, .. } => Some(report),
            _ => None,
        }
    }

    /// BIC on the validation data, if validated.
    pub fn bic(&self) -> Option<T> {
        self.report().and_then(Report::bic)
    }

    /// Whether the candidate can be selected: identified with converged iterations, stable, and
    /// passing every test.
    pub fn selectable(&self) -> bool {
        matches!(&self.outcome, Outcome::Validated { result, report } if result.converged && report.passed())
    }
}

/// Result of `search`: every candidate in the order of the structures, and the selected one.
#[derive(Clone, Debug)]
pub struct StructureSearch<T> {
    pub candidates: Vec<Candidate<T>>,
    /// Index of the selected candidate in `candidates` (`None` if no candidate is selectable).
    pub selected: Option<usize>,
}

impl<T> StructureSearch<T> {
    /// The selected candidate.
    pub fn selected(&self) -> Option<&Candidate<T>> {
        self.selected.map(|i| &self.candidates[i])
    }
}

/// Search of the model structure: for every `structure`, SRIVC on the identification data
/// `(u, y)`, simulation of the model on the validation data `(u_val, y_val)` (another experiment,
/// or the second half of the same one with `evaluated_from`), and `Validation::check` with the
/// information criteria and `options.checks` at `options.confidence`.
///
/// Selection: among the candidates with converged iterations, a stable model and every test
/// passed, the lowest BIC. The tests reject the structures that miss dynamics or have a wrong
/// delay (the residual still depends on the input); BIC rejects the ones with parameters that do
/// not improve the fit (which pass the tests as well).
///
/// The delay has to be searched together with the orders: missing poles are made up for by a
/// longer delay, so the best delay depends on the order.
///
/// The integrators `q` too: with too few, `A` needs a pole near the origin, with too many, `B` a
/// zero near the origin; either costs a parameter that does not improve the fit, so that BIC
/// picks the right `q` (from candidates with `n` and `m` adjusted). With `options.prefilter`, the
/// validation data are high-passed (the input and the output alike) before the simulation.
///
/// Errors of the validation that do not depend on the candidate (e.g. too few segments or
/// periods for the checks) are returned.
pub fn search<T>(
    identification: (&[T], &[T]),
    validation: (&[T], &[T]),
    ts: T,
    structures: &[Structure],
    options: &SearchOptions<T>,
) -> Result<StructureSearch<T>, ValidationError>
where
    T: Float + AddAssign + MulAssign + ComplexField + RealField + FftNum,
{
    let (u, y) = identification;
    let (u_val, y_val) = match options.prefilter {
        Some(cutoff) => {
            let order = structures.iter().map(|s| s.integrators).max().unwrap_or(0) + 1;
            high_pass(validation.0, validation.1, cutoff / T::from(2.0 * std::f64::consts::PI).unwrap(), ts, order)
        }
        None => (validation.0.to_vec(), validation.1.to_vec()),
    };
    let mut candidates = Vec::with_capacity(structures.len());
    for &structure in structures {
        let srivc = SrivcOptions { input_delay: structure.input_delay, ..options.srivc.clone() };
        let (n, m, q) = (structure.denominator_order, structure.numerator_order, structure.integrators);
        let identified = match options.prefilter {
            Some(cutoff) => Prefilter::new(q, cutoff).and_then(|prefilter| identify_with_prefilter(u, y, ts, n, m, &prefilter, &options.initialization, &srivc)),
            None if q == 0 => identify(u, y, ts, n, m, &options.initialization, &srivc),
            None => Err(SrivcError::NoPrefilter { integrators: q }),
        };
        let outcome = match identified {
            Err(error) => Outcome::NotIdentified(error),
            Ok(result) if result.parameter.iter().any(|v| !Float::is_finite(*v)) => Outcome::Unstable(result),
            Ok(result) => {
                let validation = Validation::continuous(&result.model, ts, &u_val, &y_val)?.evaluated_from_sample(options.evaluated_from);
                if !Float::is_finite(validation.mse()) {
                    Outcome::Unstable(result)
                } else {
                    let mut checks = vec![Check::InformationCriteria { parameters: structure.parameters() }];
                    checks.extend(options.checks.iter().cloned());
                    let report = validation.check(&checks, options.confidence)?;
                    Outcome::Validated { result, report }
                }
            }
        };
        candidates.push(Candidate { structure, outcome });
    }
    let selected = candidates
        .iter()
        .enumerate()
        .filter(|(_, c)| c.selectable())
        .min_by(|(_, a), (_, b)| a.bic().partial_cmp(&b.bic()).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i);
    Ok(StructureSearch { candidates, selected })
}

impl<T: Float + fmt::Display> fmt::Display for StructureSearch<T> {
    /// One row per candidate: structure, BIC, pass / fail of each test, SRIVC iterations; then the
    /// selected structure.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "  n  m  q  nk |          BIC | tests | iterations")?;
        for (i, c) in self.candidates.iter().enumerate() {
            let s = &c.structure;
            write!(f, "{:3} {:2} {:2} {:3} | ", s.denominator_order, s.numerator_order, s.integrators, s.input_delay)?;
            match &c.outcome {
                Outcome::NotIdentified(error) => write!(f, "not identified: {error}")?,
                Outcome::Unstable(result) => write!(f, "unstable model ({} iterations)", result.iterations)?,
                Outcome::Validated { result, report } => {
                    let bic = report.bic().map(|b| format!("{b:12.1}")).unwrap_or_else(|| format!("{:>12}", "-"));
                    let tests: String = report
                        .results
                        .iter()
                        .filter_map(CheckResult::passed)
                        .map(|p| if p { '+' } else { '-' })
                        .collect();
                    write!(f, "{bic} | {tests:5} | {}{}", result.iterations, if result.converged { "" } else { " (not converged)" })?;
                    if !c.selectable() {
                        write!(f, " rejected")?;
                    }
                }
            }
            if self.selected == Some(i) {
                write!(f, " <- selected")?;
            }
            writeln!(f)?;
        }
        match self.selected() {
            Some(c) => write!(f, "selected: {}", c.structure),
            None => write!(f, "no structure passes every test"),
        }
    }
}
