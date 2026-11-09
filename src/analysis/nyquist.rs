use std::marker::PhantomData;

use num_complex::Complex;
use num_traits::{float::FloatConst, Float, ToPrimitive};

use super::adaptive_grid::{log_step, InitialGrid, Refinement};
use super::{FrequencyResponse, FrequencyResponses, FrequencyTransferFunction, FrequencyUnit, Hz};

/// Nyquist plot over `[freq_from, freq_to)` with step `dfreq`, all given in `U` (`Hz` by default).
pub struct NyquistPlotter<T, U = Hz> {
    freq_from: T,
    dfreq: T,
    data_len: usize,
    _unit: PhantomData<U>,
}

impl <T, U> NyquistPlotter<T, U>
where
    T: Float + FloatConst + ToPrimitive + 'static,
    U: FrequencyUnit,
{
    /// `data_len = ceil((freq_to - freq_from) / dfreq)` points starting at `freq_from`, in `U`.
    pub fn new(freq_from: T, freq_to: T, dfreq: T) -> Self {
        Self {
            freq_from,
            dfreq,
            data_len: ((freq_to - freq_from) / dfreq).ceil().to_usize().unwrap_or(0),
            _unit: PhantomData,
        }
    }

    /// Calculate the vector locus of a frequency transfer function
    /// (or anything convertible into one, such as a continuous `TransferFunction`).
    /// `omega` of each point is in rad/s regardless of `U`.
    pub fn plot<G: Into<FrequencyTransferFunction<T>>>(&self, g: G) -> FrequencyResponses<T> {
        let g = g.into();
        (0..self.data_len)
            .map(|i| {
                let omega = U::to_rad_per_sec(self.freq_from + T::from(i).unwrap() * self.dfreq);
                FrequencyResponse { omega, value: g.response(omega) }
            })
            .collect()
    }
}

/// Nyquist plot over `[freq_from, freq_to]` (in `U`, `Hz` by default) with the frequencies placed
/// where the locus needs them: dense around a lightly damped mode (whose circle needs a fine step),
/// sparse where the locus is nearly straight.
///
/// The interval is first divided into `points_per_decade` intervals per decade (logarithmically;
/// uniformly into `points_per_decade` intervals if `freq_from` is 0), or into `initial_points`
/// points in all. Each interval `[a, b]` is then halved at `m` (`√(ab)`, or `(a + b) / 2` from 0)
/// while, with `G_a, G_m, G_b` the responses there,
///
/// ```text
/// |arg((G_b - G_m) / (G_m - G_a))| > max_angle      (the locus turns at m)
/// |ln(G_b / G_a)|, |ln(G_m / G_a)| or |ln(G_b / G_m)| > max_log_step
///                                                   (|G| or the phase changes much: √(Δln|G|² + Δarg²))
/// ```
///
/// unless `b / a - 1 < min_ratio` (`b - a < min_ratio * freq_to` from 0) or `max_points` are reached.
/// `G` is evaluated once per new point. The intervals are refined depth first from the lowest
/// frequency, and the right end of each interval that is not halved is appended, so the points come
/// out in increasing frequency without sorting. When `max_points` are reached, the intervals not yet
/// examined (the highest frequencies) are kept as they are.
/// A coarser initial grid does not always give fewer points: halving overshoots the step needed
/// where the gain is steep (e.g. `1/s^3` with `max_log_step` 0.1 needs a ratio of `e^0.033` between
/// points, which halving an interval of ratio 3.16 reaches only at `e^0.018`).
/// A mode narrower than the initial step can be missed when its phase comes back within the
/// interval (a resonance and an antiresonance); make the initial grid finer for such modes.
pub struct AdaptiveNyquistPlotter<T, U = Hz> {
    freq_from: T,
    freq_to: T,
    refinement: Refinement<T>,
    max_angle: T,
    max_log_step: T,
    _unit: PhantomData<U>,
}

impl <T, U> AdaptiveNyquistPlotter<T, U>
where
    T: Float + FloatConst + ToPrimitive + 'static,
    U: FrequencyUnit,
{
    /// `0 <= freq_from < freq_to`, in `U`. Defaults: `points_per_decade` 20, `max_angle` 2° (in rad),
    /// `max_log_step` 0.1, `min_ratio` 1e-6, `max_points` 100 000.
    pub fn new(freq_from: T, freq_to: T) -> Self {
        Self {
            freq_from,
            freq_to,
            refinement: Refinement::new(),
            max_angle: T::from(2.0).unwrap().to_radians(),
            max_log_step: T::from(0.1).unwrap(),
            _unit: PhantomData,
        }
    }

    /// Initial grid of `n` intervals per decade (at least 1). Replaces `initial_points`.
    pub fn points_per_decade(mut self, n: usize) -> Self {
        self.refinement.initial = InitialGrid::PerDecade(n.max(1));
        self
    }

    /// Initial grid of `n` points over the whole interval, both ends included (at least 2),
    /// whatever its number of decades. Replaces `points_per_decade`.
    pub fn initial_points(mut self, n: usize) -> Self {
        self.refinement.initial = InitialGrid::Points(n.max(2));
        self
    }

    /// Largest turn \[rad\] of the locus at the middle of an interval.
    pub fn max_angle(mut self, angle: T) -> Self {
        self.max_angle = angle;
        self
    }

    /// Largest `|ln(G_2 / G_1)|` between neighboring points.
    pub fn max_log_step(mut self, step: T) -> Self {
        self.max_log_step = step;
        self
    }

    /// Relative width `b / a - 1` below which an interval is not halved.
    pub fn min_ratio(mut self, ratio: T) -> Self {
        self.refinement.min_ratio = ratio;
        self
    }

    /// Upper bound of the number of points.
    pub fn max_points(mut self, n: usize) -> Self {
        self.refinement.max_points = n;
        self
    }

    /// Calculate the vector locus of a frequency transfer function
    /// (or anything convertible into one, such as a continuous `TransferFunction`), in increasing
    /// frequency from `freq_from` to `freq_to` (both included).
    /// `omega` of each point is in rad/s regardless of `U`.
    pub fn plot<G: Into<FrequencyTransferFunction<T>>>(&self, g: G) -> FrequencyResponses<T> {
        let from = U::to_rad_per_sec(self.freq_from);
        let to = U::to_rad_per_sec(self.freq_to);
        FrequencyResponses(self.refinement.responses(&g.into(), from, to, |a, m, b| self.needs_split(a, m, b)))
    }

    fn needs_split(&self, a: Complex<T>, m: Complex<T>, b: Complex<T>) -> bool {
        if log_step(a, b) > self.max_log_step || log_step(a, m) > self.max_log_step || log_step(m, b) > self.max_log_step {
            return true;
        }
        let (d1, d2) = (m - a, b - m);
        let scale = a.norm().max(m.norm()).max(b.norm());
        let tiny = T::epsilon() * scale;
        d1.norm() > tiny && d2.norm() > tiny && (d2 / d1).arg().abs() > self.max_angle
    }
}
