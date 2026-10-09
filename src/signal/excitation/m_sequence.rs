//! Maximum-length sequences (M-sequences).

use std::marker::PhantomData;

use nalgebra::{ComplexField, RealField};
use num_traits::Float;

use crate::{DiscreteSystem, Siso};

use super::ExcitationError;

/// Feedback taps of a maximum-length Fibonacci LFSR for the orders 2 ..= 32: the exponents of a
/// primitive polynomial over GF(2), the order first (e.g. `x^5 + x^3 + 1` for order 5). Every
/// entry was checked to give the period `2^n - 1`.
const TAPS: [&[usize]; 31] = [
    &[2, 1],
    &[3, 2],
    &[4, 3],
    &[5, 3],
    &[6, 5],
    &[7, 6],
    &[8, 6, 5, 4],
    &[9, 5],
    &[10, 7],
    &[11, 9],
    &[12, 11, 10, 4],
    &[13, 12, 11, 8],
    &[14, 13, 12, 2],
    &[15, 14],
    &[16, 15, 13, 4],
    &[17, 14],
    &[18, 11],
    &[19, 18, 17, 14],
    &[20, 17],
    &[21, 19],
    &[22, 21],
    &[23, 18],
    &[24, 23, 22, 17],
    &[25, 22],
    &[26, 6, 2, 1],
    &[27, 5, 2, 1],
    &[28, 25],
    &[29, 27],
    &[30, 6, 4, 1],
    &[31, 28],
    &[32, 22, 2, 1],
];

/// Generator of the M-sequence of order `order` (`n`, 2 ..= 32), each bit held for `clock`
/// samples: `step` gives one sample as `±1` and advances, periodically without end (period
/// `(2^n - 1) clock` samples); `step_bit` gives the bit itself (e.g. for a digital output). The
/// state is one `u64`, whatever the order; `Iterator` gives the samples of `step`, e.g. `.take(n)`
/// for a buffer of an explicit length.
///
/// ```text
/// s[k + n] = s[k + n - t_1] ⊕ s[k + n - t_2] ⊕ ...    (taps t_i of TAPS, from s = 1 ... 1)
/// step_bit j -> s[⌊j / clock⌋ mod (2^n - 1)],          j = 0, 1, 2, ...
/// step j     -> 2 s[⌊j / clock⌋ mod (2^n - 1)] - 1     (±1)
/// ```
///
/// - RMS 1 like `multisine`; the mean over a period is `1 / (2^n - 1)` (one more `+1` than `-1`).
/// - Periodic autocorrelation (`clock = 1`): `Σ_k u[k] u[k + m] = 2^n - 1` for `m ≡ 0`, `-1`
///   otherwise, i.e. a nearly white input within the period.
/// - Spectrum: lines at the multiples of `1 / ((2^n - 1) clock ts)` with the envelope
///   `sinc^2(f clock ts)`, which is down by 3 dB at about `0.44 / (clock ts)`: `clock` sets the
///   band, `order` the frequency resolution.
///
/// With whole periods the power is only at the lines, as for `multisine` (`Validation::line_test`
/// with the fundamental frequency `1 / ((2^n - 1) clock ts)`).
///
/// ```
/// use dsmc::signal::excitation::MSequence;
///
/// // Order 10 (1023 bits), each bit held for 2 samples
/// let mut ms = MSequence::<f64>::new(2, 10).unwrap();
/// assert_eq!(ms.period(), 2046);
///
/// // One sample per control period, as ±0.5 around an operating point of 1.0
/// let u = 1.0 + 0.5 * ms.step();
///
/// // Or a buffer of two periods
/// ms.reset();
/// let u: Vec<f64> = ms.by_ref().take(2 * 2046).collect();
/// assert_eq!(u.iter().filter(|&&v| v > 0.0).count(), 2 * 2 * 512);
/// ```
#[derive(Clone, Debug)]
pub struct MSequence<T> {
    order: usize,
    clock: usize,
    taps: &'static [usize],
    /// Shift register: the output is stage 1 (bit 0), the feedback enters stage n (bit n - 1).
    state: u64,
    /// Samples of the current bit given so far (`0 .. clock`).
    held: usize,
    period: usize,
    _value: PhantomData<T>,
}

impl<T: Float> MSequence<T> {
    /// Errors: `InvalidOrder` if `order` is not within 2 ..= 32, `InvalidClock` if `clock` is 0 or
    /// the period `(2^n - 1) clock` overflows `usize`.
    pub fn new(clock: usize, order: usize) -> Result<Self, ExcitationError> {
        if !(2..=32).contains(&order) {
            return Err(ExcitationError::InvalidOrder { order });
        }
        let bits = (1u64 << order) - 1;
        let period = usize::try_from(bits).ok()
            .and_then(|bits| bits.checked_mul(clock))
            .filter(|_| clock > 0).ok_or(ExcitationError::InvalidClock { clock, order })?;
        Ok(Self { order, clock, taps: TAPS[order - 2], state: bits, held: 0, period, _value: PhantomData })
    }

    /// The current sample as `±1` (`+1` for the bit 1), then advance by one sample.
    pub fn step(&mut self) -> T {
        if self.step_bit() { T::one() } else { -T::one() }
    }

    /// The current bit, then advance by one sample.
    pub fn step_bit(&mut self) -> bool {
        let bit = self.state & 1 == 1;
        self.held += 1;
        if self.held == self.clock {
            self.held = 0;
            // Fibonacci LFSR: tap t reads bit n - t
            let feedback = self.taps.iter().fold(0, |acc, &t| acc ^ ((self.state >> (self.order - t)) & 1));
            self.state = (self.state >> 1) | (feedback << (self.order - 1));
        }
        bit
    }

    /// Period `(2^n - 1) clock` \[samples\].
    pub fn period(&self) -> usize {
        self.period
    }

    /// Back to the start of the sequence (the state `1 ... 1`).
    pub fn reset(&mut self) {
        self.state = (1u64 << self.order) - 1;
        self.held = 0;
    }
}

/// Endless: `next` is always `Some(step())`.
impl<T: Float> Iterator for MSequence<T> {
    type Item = T;

    fn next(&mut self) -> Option<T> {
        Some(self.step())
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (usize::MAX, None)
    }
}

/// `MSequence` through the filter `filter` (`F(z)`): `step` feeds the next `±1` sample to the
/// filter and gives its output, so that the spectrum of the M-sequence is
/// shaped by `|F|` (e.g. more power where the plant gain is low).
///
/// ```text
/// y[k] = F(z) u[k],   u[k] = ±1 (MSequence::step)
/// ```
///
/// - Not binary any more: the crest factor (peak / RMS) is above the 1 of the M-sequence, by how
///   much depending on the filter, so that the peak of the input for a given RMS (the power at the
///   lines) is larger.
/// - The filter starts from rest: the first period contains its transient, and the output is
///   periodic with the period of the M-sequence only after it (leave it out of an evaluation).
/// - The output is not scaled: its level is that of `F` applied to `±1`.
///
/// ```
/// use dsmc::{tf, DiscreteSystem, discretize::Tustin};
/// use dsmc::signal::excitation::ShapedMSequence;
///
/// // Order 10, one bit per sample, through a first-order low-pass at 100 Hz (ts = 1 ms)
/// let w = 2.0 * std::f64::consts::PI * 100.0;
/// let filter = tf!("{w} / (s + {w})").discretize(Tustin, 1e-3).unwrap();
/// let mut ms = ShapedMSequence::new(1, 10, DiscreteSystem::try_from(&filter).unwrap()).unwrap();
/// let u: Vec<f64> = (0..2 * ms.period()).map(|_| ms.step()).collect();
/// ```
#[derive(Clone, Debug)]
pub struct ShapedMSequence<T> {
    sequence: MSequence<T>,
    filter: DiscreteSystem<T, Siso>,
}

impl<T: Float + ComplexField + RealField> ShapedMSequence<T> {
    /// The M-sequence of `MSequence::new(clock, order)` (and its errors) through `filter`, which is
    /// reset to rest.
    pub fn new(clock: usize, order: usize, filter: impl Into<DiscreteSystem<T, Siso>>) -> Result<Self, ExcitationError> {
        let sequence = MSequence::new(clock, order)?;
        let mut filter = filter.into();
        filter.reset();
        Ok(Self { sequence, filter })
    }

    /// The output of the filter for the next sample of the M-sequence.
    pub fn step(&mut self) -> T {
        self.filter.update(self.sequence.step())
    }

    /// Period of the M-sequence `(2^n - 1) clock` \[samples\] (that of the output after the
    /// transient of the filter).
    pub fn period(&self) -> usize {
        self.sequence.period()
    }

    /// Back to the start: the M-sequence to its start, the filter to rest.
    pub fn reset(&mut self) {
        self.sequence.reset();
        self.filter.reset();
    }
}
