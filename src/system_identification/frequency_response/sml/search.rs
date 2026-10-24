//! Search of the model structure `(n, m, q, delay)` of the SML estimator by the BIC.

use std::fmt;

use nalgebra::RealField;
use num_traits::Float;
use rustfft::FftNum;

use super::{SmlError, SmlIdentifier, SmlIdentifierOptions, SmlOptions, SmlResultWithDelay, Termination};

/// Model structure `e^(-delay s) B(s) / (s^q A(s))` of `SmlIdentifier::identify`:
/// `deg A = n = denominator_order`, `deg B = m = numerator_order`, `q = integrators`, `delay` \[s\],
/// always in this order `(n, m, q, delay)` (constructor, `grid`, display), as `srivc::Structure`
/// but with the delay in seconds.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SmlStructure<T> {
    /// `n`.
    pub denominator_order: usize,
    /// `m`.
    pub numerator_order: usize,
    /// `q`.
    pub integrators: usize,
    /// Dead time \[s\].
    pub delay: T,
}

impl<T: Copy> SmlStructure<T> {
    /// `(n, m, q, delay)`: e.g. `b_0 e^(-0.002 s) / (s (s^2 + a_1 s + a_2))` is
    /// `SmlStructure::new(2, 0, 1, 0.002)`.
    pub fn new(denominator_order: usize, numerator_order: usize, integrators: usize, delay: T) -> Self {
        Self { denominator_order, numerator_order, integrators, delay }
    }

    /// Number of estimated parameters `nθ = n + m + 1` (the integrators and the delay are fixed).
    pub fn parameters(&self) -> usize {
        self.denominator_order + self.numerator_order + 1
    }

    /// Every structure with `n` in `denominator_orders`, `m` in `numerator_orders` (`m <= n`), `q`
    /// in `integrators` and the dead time in `delays` \[s\] (arguments in the order
    /// `(n, m, q, delay)`).
    pub fn grid(
        denominator_orders: impl IntoIterator<Item = usize>,
        numerator_orders: impl IntoIterator<Item = usize> + Clone,
        integrators: impl IntoIterator<Item = usize> + Clone,
        delays: impl IntoIterator<Item = T> + Clone,
    ) -> Vec<Self> {
        let mut structures = Vec::new();
        for n in denominator_orders {
            for m in numerator_orders.clone().into_iter().filter(|&m| m <= n) {
                for q in integrators.clone() {
                    for delay in delays.clone() {
                        structures.push(Self::new(n, m, q, delay));
                    }
                }
            }
        }
        structures
    }
}

impl<T: Float + fmt::Display> fmt::Display for SmlStructure<T> {
    /// `(n, m, q, delay) = (2, 2, 2, 0.002)`.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "(n, m, q, delay) = ({}, {}, {}, {})", self.denominator_order, self.numerator_order, self.integrators, self.delay)
    }
}

/// Options of `SmlIdentifier::search`.
#[derive(Clone, Debug)]
pub struct SmlSearchOptions<T> {
    /// Options of the estimator of every candidate.
    pub sml: SmlOptions<T>,
    /// BIC difference below which two structures are not told apart (default 2): of the
    /// candidates within `bic_margin` of the lowest BIC, the one with the fewest parameters is
    /// selected. 0 selects the lowest BIC.
    pub bic_margin: T,
}

impl<T: Float> Default for SmlSearchOptions<T> {
    fn default() -> Self {
        Self { sml: SmlOptions::default(), bic_margin: T::from(2).unwrap() }
    }
}

/// One candidate of `SmlIdentifier::search`.
#[derive(Clone, Debug)]
pub struct SmlCandidate<T> {
    pub structure: SmlStructure<T>,
    /// The estimate, or why there is none (e.g. `TooFewLines`, `Singular`).
    pub outcome: Result<SmlResultWithDelay<T>, SmlError>,
}

impl<T: Float> SmlCandidate<T> {
    /// The estimate, if any.
    pub fn fit(&self) -> Option<&SmlResultWithDelay<T>> {
        self.outcome.as_ref().ok()
    }

    /// BIC of the estimate (`SmlResult::bic`), if any.
    pub fn bic(&self) -> Option<T> {
        self.fit().map(|fit| fit.result.bic())
    }

    /// Whether the candidate can be selected: estimated, and the iterations did not stop at
    /// `max_iterations` (`Termination::Stalled`, at a minimum within rounding, is accepted).
    pub fn selectable(&self) -> bool {
        self.fit().is_some_and(|fit| fit.result.termination != Termination::MaxIterations)
    }
}

/// Result of `SmlIdentifier::search`: every candidate in the order of the structures, and the
/// selected one.
#[derive(Clone, Debug)]
pub struct SmlSearch<T> {
    pub candidates: Vec<SmlCandidate<T>>,
    /// Index of the selected candidate in `candidates` (`None` if no candidate is selectable).
    pub selected: Option<usize>,
}

impl<T: Float> SmlSearch<T> {
    /// The selected candidate.
    pub fn selected(&self) -> Option<&SmlCandidate<T>> {
        self.selected.map(|i| &self.candidates[i])
    }

    /// The selectable candidates by increasing BIC.
    pub fn ranked(&self) -> Vec<&SmlCandidate<T>> {
        let mut ranked: Vec<&SmlCandidate<T>> = self.candidates.iter().filter(|c| c.selectable()).collect();
        ranked.sort_by(|a, b| a.bic().partial_cmp(&b.bic()).unwrap_or(std::cmp::Ordering::Equal));
        ranked
    }
}

impl<T: Float + FftNum + RealField> SmlIdentifier<T> {
    /// Search of the model structure: `identify` for every structure, and the selection by the
    /// BIC (`SmlResult::bic`, on the same data for every candidate):
    ///
    /// 1. the selectable candidates (`SmlCandidate::selectable`) within `options.bic_margin` of
    ///    the lowest BIC,
    /// 2. of them, the fewest parameters `nθ`, then the lowest BIC.
    ///
    /// A BIC difference of 1 or 2 is not significant: e.g. a pole close to the origin, below the
    /// lowest line, cannot be told from an integrator, and `q - 1` integrators with one more pole
    /// may have a slightly lower BIC than the `q` of the plant; the margin then prefers the
    /// structure with fewer parameters. The cost test (`SmlResult::cost_test`) is not used: with
    /// few periods or a low signal-to-noise ratio it passes structures with fewer poles and a much
    /// higher cost.
    ///
    /// The delay has to be searched together with the orders: missing poles are made up for by a
    /// longer delay, so the best delay depends on the order.
    pub fn search(&self, structures: &[SmlStructure<T>], options: &SmlSearchOptions<T>) -> SmlSearch<T> {
        let candidates: Vec<SmlCandidate<T>> = structures
            .iter()
            .map(|&structure| {
                let (n, m, q) = (structure.denominator_order, structure.numerator_order, structure.integrators);
                SmlCandidate { structure, outcome: self.identify(n, m, q, structure.delay, &options.sml) }
            })
            .collect();
        let lowest = candidates.iter().filter(|c| c.selectable()).filter_map(SmlCandidate::bic).fold(T::infinity(), Float::min);
        let selected = candidates
            .iter()
            .enumerate()
            .filter(|(_, c)| c.selectable() && c.bic().is_some_and(|b| b <= lowest + options.bic_margin))
            .min_by(|(_, a), (_, b)| {
                let by_parameters = a.structure.parameters().cmp(&b.structure.parameters());
                by_parameters.then(a.bic().partial_cmp(&b.bic()).unwrap_or(std::cmp::Ordering::Equal))
            })
            .map(|(i, _)| i);
        SmlSearch { candidates, selected }
    }
}

/// `SmlIdentifier::new` and `SmlIdentifier::search` in one call.
pub fn search<T: Float + FftNum + RealField>(
    u: &[T],
    y: &[T],
    ts: T,
    structures: &[SmlStructure<T>],
    record: &SmlIdentifierOptions<T>,
    options: &SmlSearchOptions<T>,
) -> Result<SmlSearch<T>, SmlError> {
    Ok(SmlIdentifier::new(u, y, ts, record)?.search(structures, options))
}

impl<T: Float + fmt::Display> fmt::Display for SmlSearch<T> {
    /// One row per candidate: structure (the delay in ms), cost with its expected value and standard deviation, cost
    /// test at 2 standard deviations, BIC, iterations; then the selected structure.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "  n  m  q  delay[ms] |         cost    expected ±    std | test |          BIC | iterations")?;
        for (i, c) in self.candidates.iter().enumerate() {
            let s = &c.structure;
            write!(f, "{:3} {:2} {:2} {:10.3} | ", s.denominator_order, s.numerator_order, s.integrators, s.delay * T::from(1000).unwrap())?;
            match &c.outcome {
                Err(error) => write!(f, "not identified: {error}")?,
                Ok(fit) => {
                    let r = &fit.result;
                    let test = if r.cost_test(T::from(2).unwrap()) { "pass" } else { "fail" };
                    write!(f, "{:12.1} {:11.1} ± {:6.1} | {test} | {:12.1} | {}", r.cost, r.expected_cost, r.cost_std, r.bic(), r.iterations)?;
                    if r.termination == Termination::MaxIterations {
                        write!(f, " (not converged) rejected")?;
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
            None => write!(f, "no structure identified"),
        }
    }
}
