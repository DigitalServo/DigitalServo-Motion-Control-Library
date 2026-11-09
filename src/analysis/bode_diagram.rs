use std::borrow::Borrow;
use std::marker::PhantomData;

use num_complex::Complex;
use num_traits::{float::FloatConst, Float, ToPrimitive};

use crate::{Continuous, Discrete, TransferFunction};

use super::adaptive_grid::{log_step, InitialGrid, Refinement};
use super::{FrequencyCharacteristics, FrequencyTransferFunction, FrequencyUnit, Hz};

/// Bode diagram over `[freq_from, freq_to)` with step `dfreq`, all given in `U` (`Hz` by default).
pub struct BodeDiagramPlotter<T, U = Hz> {
    freq_from: T,
    dfreq: T,
    data_len: usize,
    log_scale: bool,
    unwrap_phase: bool,
    _unit: PhantomData<U>,
}

impl <T, U> BodeDiagramPlotter<T, U>
where
    T: Float + FloatConst + ToPrimitive + 'static,
    U: FrequencyUnit,
{
    /// `data_len = ceil((freq_to - freq_from) / dfreq)` points starting at `freq_from`, in `U`.
    /// Gain in dB if `log_scale`, otherwise the magnitude. Phase unwrapped (see `unwrap_phase`).
    pub fn new(freq_from: T, freq_to: T, dfreq: T, log_scale: bool) -> Self {
        Self {
            freq_from,
            dfreq,
            data_len: ((freq_to - freq_from) / dfreq).ceil().to_usize().unwrap_or(0),
            log_scale,
            unwrap_phase: true,
            _unit: PhantomData,
        }
    }

    /// If `unwrap` (the default), make the phase continuous along the frequency axis (e.g. for a dead time,
    /// whose phase decreases without bound); otherwise wrap it into `(-π, π]`. Each point is shifted by a multiple of 2π
    /// so that its step from the previous point lies in `(-π, π]`. The first point stays in `(-π, π]`.
    /// The step `dfreq` must be fine enough that the true phase changes by less than π between points.
    pub fn unwrap_phase(mut self, unwrap: bool) -> Self {
        self.unwrap_phase = unwrap;
        self
    }

    /// Calculate the frequency characteristics of a frequency transfer function
    /// (or anything convertible into one, such as a continuous `TransferFunction`).
    pub fn plot<G: Into<FrequencyTransferFunction<T>>>(&self, g: G) -> Vec<FrequencyCharacteristics<T, U>> {
        let g = g.into();
        let mut points: Vec<FrequencyCharacteristics<T, U>> = (0..self.data_len)
            .map(|i| g.characteristics(self.freq_from + T::from(i).unwrap() * self.dfreq, self.log_scale))
            .collect();
        if self.unwrap_phase {
            unwrap_phase(&mut points);
        }
        points
    }

    /// Calculate the frequency response of a polynomial in the s-domain.
    /// If polynomial is an*s^n +an-1*s^(n-1) + ... + a0, coefficient should be set \[an, an-1, ..., a0\]
    pub fn frequency_response_s<S: Borrow<TransferFunction<T, Continuous>>>(&self, tf: S) -> Vec<FrequencyCharacteristics<T, U>> {
        self.plot(tf.borrow().frequency_transfer_function())
    }

    /// Calculate the frequency response of a polynomial in the z-domain.
    /// If polynomial is an*z^n +an-1*z^(n-1) + ... + a0, coefficient should be set \[an, an-1, ..., a0\]
    pub fn frequency_response_z<S: Borrow<TransferFunction<T, Discrete>>>(&self, tf: S,  ts: T) -> Vec<FrequencyCharacteristics<T, U>> {
        self.plot(tf.borrow().frequency_transfer_function(ts))
    }
}

/// Bode diagram over `[freq_from, freq_to]` (in `U`, `Hz` by default) with the frequencies placed
/// where the curves need them: dense around a lightly damped mode, sparse where the gain and the
/// phase are nearly straight lines against the logarithmic frequency.
///
/// The interval is first divided into `points_per_decade` intervals per decade (logarithmically;
/// uniformly into `points_per_decade` intervals if `freq_from` is 0), or into `initial_points`
/// points in all. Each interval `[a, b]` is then halved at `m = √(ab)` (`(a + b) / 2` from 0),
/// the middle on the logarithmic axis, while, with `G_a, G_m, G_b` the responses there and
/// `ln G = ln|G| + j arg G` (the gain in nepers and the phase in rad together),
///
/// ```text
/// |ln(G_b / G_a)|, |ln(G_m / G_a)| or |ln(G_b / G_m)| > max_log_step
///                                       (the gain or the phase changes much: √(Δln|G|² + Δarg²))
/// |ln(G_m / G_a) - ln(G_b / G_m)| / 2 > max_deviation
///                                       (G_m off the straight line from G_a to G_b, in ln G)
/// ```
///
/// unless `b / a - 1 < min_ratio` (`b - a < min_ratio * freq_to` from 0) or `max_points` are reached.
/// `G` is evaluated once per new point. The intervals are refined depth first from the lowest
/// frequency, and the right end of each interval that is not halved is appended, so the points come
/// out in increasing frequency without sorting. When `max_points` are reached, the intervals not yet
/// examined (the highest frequencies) are kept as they are.
/// 1 Np = 8.69 dB: the defaults bound the step to 0.87 dB / 5.7° and the deviation to 0.087 dB / 0.57°.
/// A coarser initial grid does not always give fewer points: halving overshoots the step needed
/// where the gain is steep (e.g. `1/s^3` with `max_log_step` 0.1 needs a ratio of `e^0.033` between
/// points, which halving an interval of ratio 3.16 reaches only at `e^0.018`).
/// A mode narrower than the initial step can be missed when its phase comes back within the
/// interval (a resonance and an antiresonance); make the initial grid finer for such modes.
pub struct AdaptiveBodeDiagramPlotter<T, U = Hz> {
    freq_from: T,
    freq_to: T,
    log_scale: bool,
    unwrap_phase: bool,
    refinement: Refinement<T>,
    max_log_step: T,
    max_deviation: T,
    _unit: PhantomData<U>,
}

impl <T, U> AdaptiveBodeDiagramPlotter<T, U>
where
    T: Float + FloatConst + ToPrimitive + 'static,
    U: FrequencyUnit,
{
    /// `0 <= freq_from < freq_to`, in `U`. Gain in dB if `log_scale`, otherwise the magnitude.
    /// Defaults: `points_per_decade` 20, `max_log_step` 0.1, `max_deviation` 0.01, `min_ratio` 1e-6,
    /// `max_points` 100 000, phase unwrapped.
    pub fn new(freq_from: T, freq_to: T, log_scale: bool) -> Self {
        Self {
            freq_from,
            freq_to,
            log_scale,
            unwrap_phase: true,
            refinement: Refinement::new(),
            max_log_step: T::from(0.1).unwrap(),
            max_deviation: T::from(0.01).unwrap(),
            _unit: PhantomData,
        }
    }

    /// As `BodeDiagramPlotter::unwrap_phase`. The phase steps are below `max_log_step`, so the
    /// unwrapping holds unless the refinement stops early (`min_ratio`, `max_points`).
    pub fn unwrap_phase(mut self, unwrap: bool) -> Self {
        self.unwrap_phase = unwrap;
        self
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

    /// Largest `|ln(G_2 / G_1)|` between neighboring points.
    pub fn max_log_step(mut self, step: T) -> Self {
        self.max_log_step = step;
        self
    }

    /// Largest deviation of `ln G` at the middle of an interval from the straight line through its ends.
    pub fn max_deviation(mut self, deviation: T) -> Self {
        self.max_deviation = deviation;
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

    /// Calculate the frequency characteristics of a frequency transfer function
    /// (or anything convertible into one, such as a continuous `TransferFunction`), in increasing
    /// frequency from `freq_from` to `freq_to` (both included).
    pub fn plot<G: Into<FrequencyTransferFunction<T>>>(&self, g: G) -> Vec<FrequencyCharacteristics<T, U>> {
        let from = U::to_rad_per_sec(self.freq_from);
        let to = U::to_rad_per_sec(self.freq_to);
        let responses = self.refinement.responses(&g.into(), from, to, |a, m, b| self.needs_split(a, m, b));
        let last = responses.len().saturating_sub(1);
        let mut points: Vec<FrequencyCharacteristics<T, U>> = responses
            .iter()
            .enumerate()
            .map(|(i, r)| {
                // The ends exactly as given, not converted back from rad/s
                let frequency = if i == 0 { self.freq_from } else if i == last { self.freq_to } else { U::from_rad_per_sec(r.omega) };
                let gain = if self.log_scale { T::from(20.0).unwrap() * r.value.norm().log10() } else { r.value.norm() };
                FrequencyCharacteristics { frequency, gain, phase: r.value.arg(), _unit: PhantomData }
            })
            .collect();
        if self.unwrap_phase {
            unwrap_phase(&mut points);
        }
        points
    }

    fn needs_split(&self, a: Complex<T>, m: Complex<T>, b: Complex<T>) -> bool {
        let (first, second) = ((m / a).ln(), (b / m).ln());
        log_step(a, b) > self.max_log_step
            || first.norm() > self.max_log_step
            || second.norm() > self.max_log_step
            || (first - second).norm() / (T::one() + T::one()) > self.max_deviation
    }
}

/// Shift each phase by a multiple of 2π so that its step from the previous (unwrapped) phase lies in `(-π, π]`.
fn unwrap_phase<T: Float + FloatConst, U>(points: &mut [FrequencyCharacteristics<T, U>]) {
    let two_pi = T::TAU();
    for i in 1..points.len() {
        let step = points[i].phase - points[i - 1].phase;
        let turns = ((step + T::PI()) / two_pi).ceil() - T::one();
        points[i].phase = points[i].phase - turns * two_pi;
    }
}
