use std::marker::PhantomData;

use num_complex::Complex;
use num_traits::{float::FloatConst, Float, ToPrimitive};

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
/// uniformly into `points_per_decade` intervals if `freq_from` is 0). Each interval `[a, b]` is then
/// halved at `m` (`√(ab)`, or `(a + b) / 2` from 0) while, with `G_a, G_m, G_b` the responses there,
///
/// ```text
/// |arg((G_b - G_m) / (G_m - G_a))| > max_angle      (the locus turns at m)
/// |ln(G_b / G_a)|, |ln(G_m / G_a)| or |ln(G_b / G_m)| > max_log_step
///                                                   (|G| or the phase changes much: √(Δln|G|² + Δarg²))
/// ```
///
/// unless `b / a - 1 < min_ratio` (`b - a < min_ratio * freq_to` from 0) or `max_points` are reached
/// (the intervals left are then kept as they are, from the highest frequencies).
/// A mode narrower than the initial step can be missed when its phase comes back within the
/// interval (a resonance and an antiresonance); raise `points_per_decade` for such modes.
pub struct AdaptiveNyquistPlotter<T, U = Hz> {
    freq_from: T,
    freq_to: T,
    points_per_decade: usize,
    max_angle: T,
    max_log_step: T,
    min_ratio: T,
    max_points: usize,
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
            points_per_decade: 20,
            max_angle: T::from(2.0).unwrap().to_radians(),
            max_log_step: T::from(0.1).unwrap(),
            min_ratio: T::from(1e-6).unwrap(),
            max_points: 100_000,
            _unit: PhantomData,
        }
    }

    /// Intervals per decade of the initial grid (at least 1).
    pub fn points_per_decade(mut self, n: usize) -> Self {
        self.points_per_decade = n.max(1);
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
        self.min_ratio = ratio;
        self
    }

    /// Upper bound of the number of points.
    pub fn max_points(mut self, n: usize) -> Self {
        self.max_points = n;
        self
    }

    /// Calculate the vector locus of a frequency transfer function
    /// (or anything convertible into one, such as a continuous `TransferFunction`), in increasing
    /// frequency from `freq_from` to `freq_to` (both included).
    /// `omega` of each point is in rad/s regardless of `U`.
    pub fn plot<G: Into<FrequencyTransferFunction<T>>>(&self, g: G) -> FrequencyResponses<T> {
        let g = g.into();
        let point = |omega: T| FrequencyResponse { omega, value: g.response(omega) };
        let from = U::to_rad_per_sec(self.freq_from);
        let to = U::to_rad_per_sec(self.freq_to);
        if !(from >= T::zero() && to > from) {
            return FrequencyResponses::default();
        }

        let grid: Vec<T> = if from > T::zero() {
            let n = ((to / from).log10() * T::from(self.points_per_decade).unwrap()).ceil().to_usize().unwrap_or(1).max(1);
            let ratio = (to / from).ln() / T::from(n).unwrap();
            (0..=n).map(|i| if i == n { to } else { from * (ratio * T::from(i).unwrap()).exp() }).collect()
        } else {
            let n = self.points_per_decade;
            (0..=n).map(|i| if i == n { to } else { to * T::from(i).unwrap() / T::from(n).unwrap() }).collect()
        };

        // Intervals still to examine, the lowest on top; each output point is final once pushed.
        let mut stack: Vec<(FrequencyResponse<T>, FrequencyResponse<T>)> = Vec::new();
        let initial: Vec<FrequencyResponse<T>> = grid.into_iter().map(point).collect();
        for w in initial.windows(2).rev() {
            stack.push((w[0], w[1]));
        }
        let mut points = vec![initial[0]];
        let min_width = self.min_ratio * to;
        while let Some((a, b)) = stack.pop() {
            let narrow = if a.omega > T::zero() { b.omega / a.omega - T::one() < self.min_ratio } else { b.omega - a.omega < min_width };
            if !narrow && points.len() + stack.len() + 1 < self.max_points {
                let omega_m = if a.omega > T::zero() { (a.omega * b.omega).sqrt() } else { (a.omega + b.omega) / (T::one() + T::one()) };
                let m = point(omega_m);
                if self.needs_split(a.value, m.value, b.value) {
                    stack.push((m, b));
                    stack.push((a, m));
                    continue;
                }
            }
            points.push(b);
        }
        FrequencyResponses(points)
    }

    fn needs_split(&self, a: Complex<T>, m: Complex<T>, b: Complex<T>) -> bool {
        let log_step = |x: Complex<T>, y: Complex<T>| (y / x).ln().norm();
        if log_step(a, b) > self.max_log_step || log_step(a, m) > self.max_log_step || log_step(m, b) > self.max_log_step {
            return true;
        }
        let (d1, d2) = (m - a, b - m);
        let scale = a.norm().max(m.norm()).max(b.norm());
        let tiny = T::epsilon() * scale;
        d1.norm() > tiny && d2.norm() > tiny && (d2 / d1).arg().abs() > self.max_angle
    }
}
