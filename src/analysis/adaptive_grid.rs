//! Adaptive frequency grid shared by `AdaptiveNyquistPlotter` and `AdaptiveBodeDiagramPlotter`.

use num_complex::Complex;
use num_traits::{Float, FloatConst};

use super::{FrequencyResponse, FrequencyTransferFunction};

/// Initial grid before the refinement.
#[derive(Clone, Copy, Debug)]
pub(super) enum InitialGrid {
    /// `n` intervals per decade (logarithmically); `n` intervals in all from 0.
    PerDecade(usize),
    /// `n` points in all, both ends included (logarithmically; uniformly from 0).
    Points(usize),
}

/// Common settings of the adaptive plotters.
#[derive(Clone, Copy, Debug)]
pub(super) struct Refinement<T> {
    pub initial: InitialGrid,
    pub min_ratio: T,
    pub max_points: usize,
}

impl<T: Float + FloatConst + 'static> Refinement<T> {
    pub fn new() -> Self {
        Self { initial: InitialGrid::PerDecade(20), min_ratio: T::from(1e-6).unwrap(), max_points: 100_000 }
    }

    /// Responses over `[from, to]` [rad/s], both included, in increasing frequency. Each interval
    /// `[a, b]` of the initial grid is halved at `m` (`√(ab)`, or `(a + b) / 2` from 0) while
    /// `needs_split(G_a, G_m, G_b)`, unless `b / a - 1 < min_ratio` (`b - a < min_ratio * to` from 0)
    /// or `max_points` are reached (the intervals left, the highest, are kept as they are).
    /// Empty unless `0 <= from < to`.
    pub fn responses(
        &self,
        g: &FrequencyTransferFunction<T>,
        from: T,
        to: T,
        needs_split: impl Fn(Complex<T>, Complex<T>, Complex<T>) -> bool,
    ) -> Vec<FrequencyResponse<T>> {
        let point = |omega: T| FrequencyResponse { omega, value: g.response(omega) };
        if !(from >= T::zero() && to > from) {
            return Vec::new();
        }

        let grid: Vec<T> = if from > T::zero() {
            let n = match self.initial {
                InitialGrid::PerDecade(n) => ((to / from).log10() * T::from(n).unwrap()).ceil().to_usize().unwrap_or(1),
                InitialGrid::Points(n) => n.saturating_sub(1),
            }
            .max(1);
            let ratio = (to / from).ln() / T::from(n).unwrap();
            (0..=n).map(|i| if i == n { to } else { from * (ratio * T::from(i).unwrap()).exp() }).collect()
        } else {
            let n = match self.initial {
                InitialGrid::PerDecade(n) => n,
                InitialGrid::Points(n) => n.saturating_sub(1),
            }
            .max(1);
            (0..=n).map(|i| if i == n { to } else { to * T::from(i).unwrap() / T::from(n).unwrap() }).collect()
        };

        // Intervals still to examine, the lowest on top; each output point is final once pushed.
        let initial: Vec<FrequencyResponse<T>> = grid.into_iter().map(point).collect();
        let mut stack: Vec<(FrequencyResponse<T>, FrequencyResponse<T>)> = initial.windows(2).rev().map(|w| (w[0], w[1])).collect();
        let mut points = vec![initial[0]];
        let min_width = self.min_ratio * to;
        while let Some((a, b)) = stack.pop() {
            let narrow = if a.omega > T::zero() { b.omega / a.omega - T::one() < self.min_ratio } else { b.omega - a.omega < min_width };
            if !narrow && points.len() + stack.len() + 1 < self.max_points {
                let omega_m = if a.omega > T::zero() { (a.omega * b.omega).sqrt() } else { (a.omega + b.omega) / (T::one() + T::one()) };
                let m = point(omega_m);
                if needs_split(a.value, m.value, b.value) {
                    stack.push((m, b));
                    stack.push((a, m));
                    continue;
                }
            }
            points.push(b);
        }
        points
    }
}

/// `|ln(y / x)| = √(Δln|G|² + Δarg²)` (Δarg in `(-π, π]`).
pub(super) fn log_step<T: Float>(x: Complex<T>, y: Complex<T>) -> T {
    (y / x).ln().norm()
}
