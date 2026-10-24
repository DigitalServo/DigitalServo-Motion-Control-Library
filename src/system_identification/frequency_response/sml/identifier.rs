//! `SmlIdentifier`: the SML data of a raw record, prepared once, and the models identified from it.

use nalgebra::RealField;
use num_traits::Float;
use rustfft::FftNum;

use super::{
    excited_lines, identify_continuous_with_delay, identify_discrete, line_costs_discrete, line_costs_with_delay, steady_periods, ExcitationOptions, SmlData,
    SmlError, SmlOptions, SmlResult, SmlResultWithDelay,
};
use crate::signal::detect_period;
use crate::{Discrete, TransferFunction};

/// Options of `SmlIdentifier::new`.
#[derive(Clone, Debug)]
pub struct SmlIdentifierOptions<T> {
    /// Samples per period, or `None` (default) to detect it from the input by
    /// `signal::detect_period`, up to `len / (transient_periods + 4)` samples (so that 4 periods
    /// remain).
    pub period: Option<usize>,
    /// Periods left out at the start of the record, the transient from rest (default 1).
    pub transient_periods: usize,
    /// Detection of the excited lines (`excited_lines`).
    pub excitation: ExcitationOptions<T>,
    /// Whether the input is applied through a zero-order hold (default `true`): the
    /// continuous-time models are identified on the lines with the hold compensated
    /// (`SmlData::compensate_zero_order_hold`), the discrete-time ones on the measured lines.
    pub zero_order_hold: bool,
    /// Band `[low, high]` \[Hz\] of the lines used, or `None` (default) for all the excited lines.
    pub band: Option<(T, T)>,
}

impl<T: Float> Default for SmlIdentifierOptions<T> {
    fn default() -> Self {
        Self { period: None, transient_periods: 1, excitation: ExcitationOptions::default(), zero_order_hold: true, band: None }
    }
}

/// SML identification of a periodic record: the data prepared once by `new`, then a model per
/// structure by `identify` (continuous time, with integrators and a dead time) or
/// `identify_discrete`, to compare by `SmlResult::bic`, or the structure searched by `search`.
///
/// `new` does, in order:
///
/// 1. the period: `options.period`, or detected from the input (the input of a periodic excitation
///    has no transient, so the whole record is used),
/// 2. the steady-state periods: the first `transient_periods` and the incomplete one at the end left
///    out (at least 4 must remain),
/// 3. the excited lines of the input (`excited_lines`, DC left out) and their SML data,
/// 4. the lines in `options.band`,
/// 5. for the continuous-time models, the zero-order hold compensated (if `zero_order_hold`).
///
/// ```
/// use dsmc::signal::excitation::multisine;
/// use dsmc::system_identification::frequency_response::sml::{SmlIdentifier, SmlIdentifierOptions, SmlOptions};
/// use dsmc::{TransferFunction, TransferFunctionWithDelay};
///
/// // 1000 e^(-0.002 s) / (s (s + 50)) behind a zero-order hold at 1 kHz, a multisine of 512
/// // samples on lines 1 ..= 60, 1 transient + 6 periods, 1 % noise on the output
/// let plant = TransferFunctionWithDelay::new(TransferFunction::continuous(&[1000.0], &[1.0, 50.0, 0.0]), 0.002);
/// let lines: Vec<usize> = (1..=60).collect();
/// let u: Vec<f64> = multisine(7.0 * 0.512, 1e-3, 1.0 / 0.512, &lines, |_| 1.0, 1).unwrap();
/// let y = plant.simulate(1e-3, &u).unwrap();
/// let y: Vec<f64> = y.iter().enumerate().map(|(k, v)| v + 1e-2 * ((k * 7919 % 1000) as f64 / 500.0 - 1.0)).collect();
///
/// let identifier = SmlIdentifier::new(&u, &y, 1e-3, &SmlIdentifierOptions::default()).unwrap();
/// assert_eq!((identifier.period(), identifier.excited_lines().len()), (512, 60));
/// let fit = identifier.identify(1, 0, 1, 0.002, &SmlOptions::default()).unwrap();
/// let a = &fit.model.tf.denominator;
/// assert!((a[1] - 50.0).abs() < 0.5, "{:?}", fit.model);
/// ```
#[derive(Clone, Debug)]
pub struct SmlIdentifier<T> {
    ts: T,
    period: usize,
    excited_lines: Vec<usize>,
    measured: SmlData<T>,
    continuous: SmlData<T>,
}

impl<T: Float + FftNum + RealField> SmlIdentifier<T> {
    /// The data of the input / output record `u`, `y` sampled with period `ts` \[s\] (see the
    /// steps above). Errors: `LengthMismatch`, `NoPeriod` (none detected), `TooFewPeriods`,
    /// `NoExcitedLine`, `InvalidSamplingPeriod`, `InvalidSpectralLine` (e.g. noise-free data).
    pub fn new(u: &[T], y: &[T], ts: T, options: &SmlIdentifierOptions<T>) -> Result<Self, SmlError> {
        if u.len() != y.len() {
            return Err(SmlError::LengthMismatch { input: u.len(), output: y.len() });
        }
        let period = match options.period {
            Some(period) => period,
            None => detect_period(u, u.len() / (options.transient_periods + 4)).ok_or(SmlError::NoPeriod)?,
        };
        let range = steady_periods(u, y, period, options.transient_periods)?;
        let excited_lines = excited_lines(&u[range.clone()], period, &options.excitation);
        if excited_lines.is_empty() {
            return Err(SmlError::NoExcitedLine);
        }
        let mut measured = SmlData::from_periods(&u[range.clone()], &y[range], ts, period, &excited_lines)?;
        if let Some((low, high)) = options.band {
            measured = measured.in_band(low, high);
        }
        let continuous = if options.zero_order_hold { measured.compensate_zero_order_hold(ts) } else { measured.clone() };
        Ok(Self { ts, period, excited_lines, measured, continuous })
    }

    /// Sampling period \[s\].
    pub fn ts(&self) -> T {
        self.ts
    }

    /// Samples per period (given or detected).
    pub fn period(&self) -> usize {
        self.period
    }

    /// Excited lines `k` of the input (before the band is applied).
    pub fn excited_lines(&self) -> &[usize] {
        &self.excited_lines
    }

    /// Lines as measured (in the band): the data of the discrete-time models, and the measured
    /// response `SpectralLine::response` of the sampled system.
    pub fn data(&self) -> &SmlData<T> {
        &self.measured
    }

    /// Lines of the continuous-time models: `data` with the zero-order hold compensated (if
    /// `zero_order_hold`), the response of the plant behind the hold.
    pub fn continuous_data(&self) -> &SmlData<T> {
        &self.continuous
    }

    /// `identify_continuous_with_delay` on `continuous_data`: `e^(-delay s) B(s) / (s^q A(s))` with
    /// `n = denom_order`, `m = numer_order`, `q = integrators`.
    pub fn identify(&self, denom_order: usize, numer_order: usize, integrators: usize, delay: T, options: &SmlOptions<T>) -> Result<SmlResultWithDelay<T>, SmlError> {
        identify_continuous_with_delay(&self.continuous, denom_order, numer_order, integrators, delay, options)
    }

    /// `identify_discrete` on `data`: `z^-nk B(z^-1) / A(z^-1)` with `n = denom_order`,
    /// `m = numer_order`, `nk = input_delay` (exact for a plant behind a zero-order hold).
    pub fn identify_discrete(&self, denom_order: usize, numer_order: usize, input_delay: usize, options: &SmlOptions<T>) -> Result<SmlResult<T, Discrete>, SmlError> {
        identify_discrete(&self.measured, self.ts, denom_order, numer_order, input_delay, options)
    }

    /// Cost of each line of a model of `identify` (`line_costs_with_delay` on `continuous_data`).
    pub fn line_costs(&self, fit: &SmlResultWithDelay<T>) -> Vec<T> {
        line_costs_with_delay(&self.continuous, &fit.model)
    }

    /// Cost of each line of a model of `identify_discrete` (`line_costs_discrete` on `data`).
    pub fn line_costs_discrete(&self, model: &TransferFunction<T, Discrete>) -> Vec<T> {
        line_costs_discrete(&self.measured, self.ts, model)
    }
}

/// `SmlIdentifier::new` and `SmlIdentifier::identify` in one call: the model
/// `e^(-delay s) B(s) / (s^q A(s))` (`n = denom_order`, `m = numer_order`, `q = integrators`) of
/// the record `u`, `y` (sampling period `ts` \[s\]) prepared by `record`. To compare several
/// structures on the same record, build the `SmlIdentifier` once (or use `search`).
///
/// ```
/// use dsmc::signal::excitation::multisine;
/// use dsmc::system_identification::frequency_response::sml::{self, SmlIdentifierOptions, SmlOptions};
/// use dsmc::{TransferFunction, TransferFunctionWithDelay};
///
/// // 1000 e^(-0.002 s) / (s (s + 50)) behind a zero-order hold at 1 kHz
/// let plant = TransferFunctionWithDelay::new(TransferFunction::continuous(&[1000.0], &[1.0, 50.0, 0.0]), 0.002);
/// let lines: Vec<usize> = (1..=60).collect();
/// let u: Vec<f64> = multisine(7.0 * 0.512, 1e-3, 1.0 / 0.512, &lines, |_| 1.0, 1).unwrap();
/// let y = plant.simulate(1e-3, &u).unwrap();
/// let y: Vec<f64> = y.iter().enumerate().map(|(k, v)| v + 1e-2 * ((k * 7919 % 1000) as f64 / 500.0 - 1.0)).collect();
///
/// let fit = sml::identify(&u, &y, 1e-3, 1, 0, 1, 0.002, &SmlIdentifierOptions::default(), &SmlOptions::default()).unwrap();
/// assert!((fit.model.tf.denominator[1] - 50.0).abs() < 0.5, "{:?}", fit.model);
/// ```
#[allow(clippy::too_many_arguments)]
pub fn identify<T: Float + FftNum + RealField>(
    u: &[T],
    y: &[T],
    ts: T,
    denom_order: usize,
    numer_order: usize,
    integrators: usize,
    delay: T,
    record: &SmlIdentifierOptions<T>,
    options: &SmlOptions<T>,
) -> Result<SmlResultWithDelay<T>, SmlError> {
    SmlIdentifier::new(u, y, ts, record)?.identify(denom_order, numer_order, integrators, delay, options)
}
