//! System analysis tools.

use std::{marker::PhantomData, ops::Deref};

use nalgebra::Complex;

mod bode_diagram;
mod frequency_transfer_function;
mod frequency_unit;
mod nyquist;
mod statistics;
pub mod fft;

pub use bode_diagram::BodeDiagramPlotter;
pub use frequency_transfer_function::FrequencyTransferFunction;
pub use frequency_unit::{FrequencyUnit, Hz, RadPerSec};
pub use nyquist::NyquistPlotter;
pub use statistics::Statistics;

use num_traits::{Float, FloatConst};
use serde::{ser::SerializeStruct, Serialize, Serializer};
use thiserror::Error;

/// Complex frequency response `value = G(jω)` at angular frequency `omega` [rad/s].
#[derive(Copy, Clone, Debug)]
pub struct FrequencyResponse<T> {
    /// Angular frequency [rad/s].
    pub omega: T,
    /// `G(jω)`.
    pub value: Complex<T>
}

/// Serialized flat as `[omega, re, im]`.
impl<T: Serialize> Serialize for FrequencyResponse<T> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut state = serializer.serialize_struct("FrequencyResponse", 3)?;
        state.serialize_field("omega", &self.omega)?;
        state.serialize_field("re", &self.value.re)?;
        state.serialize_field("im", &self.value.im)?;
        state.end()
    }
}

/// Frequency response `value = G(jω)` at angular frequency `omega` [rad/s] with its coherence (from
/// [`welch`](fft::welch)).
#[derive(Copy, Clone, Debug)]
pub struct FrequencyResponseWithCoherence<T> {
    /// Angular frequency [rad/s].
    pub omega: T,
    /// `G(jω)`.
    pub value: Complex<T>,
    /// Coherence `|P_yu|^2 / (P_uu P_yy)`, in `[0, 1]`.
    pub coherence: T,
}

/// The response without the coherence.
impl<T> From<FrequencyResponseWithCoherence<T>> for FrequencyResponse<T> {
    fn from(val: FrequencyResponseWithCoherence<T>) -> Self {
        FrequencyResponse { omega: val.omega, value: val.value }
    }
}

/// Serialized flat as `[omega, re, im, coherence]`.
impl<T: Serialize> Serialize for FrequencyResponseWithCoherence<T> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut state = serializer.serialize_struct("FrequencyResponseWithCoherence", 4)?;
        state.serialize_field("omega", &self.omega)?;
        state.serialize_field("re", &self.value.re)?;
        state.serialize_field("im", &self.value.im)?;
        state.serialize_field("coherence", &self.coherence)?;
        state.end()
    }
}

/// Response of the zero-order hold of the sampling period `ts` \[s\] relative to its gain `ts`,
/// `h(ω) = e^(-jω ts / 2) sinc(ω ts / 2)` (`sinc x = sin x / x`, 1 at `x = 0`): the half-sample
/// delay and the droop of the held input, `2/π` at the Nyquist frequency `π / ts`. A plant `G(s)`
/// behind the hold, sampled, has the response `G(jω) h(ω)` up to aliasing (see
/// `FrequencyResponses::compensate_zero_order_hold`).
pub fn zero_order_hold_response<T: Float>(omega: T, ts: T) -> Complex<T> {
    let x = omega * ts / (T::one() + T::one());
    let sinc = if x == T::zero() { T::one() } else { x.sin() / x };
    Complex::new(T::zero(), -x).exp() * sinc
}

/// Frequency responses, one per frequency (e.g. from
/// [`FrequencyResponsesWithCoherence::coherent_bins`], or computed from a model). Derefs to the
/// slice of the responses (`len`, `iter`, indexing, and `&responses` where a
/// `&[FrequencyResponse<T>]` is expected, e.g. by the identification methods in
/// [`system_identification::frequency_response`](crate::system_identification::frequency_response)),
/// converts from and into `Vec<FrequencyResponse<T>>` with `into`, and collects from an iterator.
/// Serialized as the sequence.
#[derive(Clone, Debug, Serialize)]
#[serde(transparent)]
pub struct FrequencyResponses<T>(pub Vec<FrequencyResponse<T>>);

/// No responses.
impl<T> Default for FrequencyResponses<T> {
    fn default() -> Self {
        Self(Vec::new())
    }
}

impl<T: Float> FrequencyResponses<T> {
    /// The measured response of a plant behind a zero-order hold of the sampling period `ts` \[s\]
    /// (e.g. [`welch`](fft::welch) of the held input and the sampled output) divided by the hold
    /// `h(ω)` ([`zero_order_hold_response`]): `Ĝ(jω) / h(ω)`, the response of the plant itself, to
    /// fit by a continuous-time model. Without it a rational model has to fit the half-sample
    /// delay `e^(-jω ts / 2)` as well, which it cannot (0.46 rad at 146 Hz with `ts` = 1 ms).
    ///
    /// Aliasing is not compensated: the result is `G(jω)` where `|G|` is small beyond the Nyquist
    /// frequency compared with the band. The frequencies must be below `2π / ts` (the first zero of
    /// `h`), i.e. within the Nyquist frequency `π / ts` of the sampled data; `ts > 0`.
    ///
    /// Weights: the standard deviation of `Ĝ / h` is that of `Ĝ` over `|h|`, so weights `1 / σ`
    /// given to a weighted fit (`levy::sanathanan_koerner_weighted`,
    /// `vector_fitting::identify_weighted`) are to be multiplied by `|h(ω)|`. The relative-error
    /// fits need no change.
    ///
    /// ```
    /// use dsmc::discretize::Zoh;
    /// use dsmc::{FrequencyResponse, FrequencyResponses, TransferFunction};
    ///
    /// // The exact ZOH equivalent of 1000 / (s^2 + 20 s + 1000) at 1 kHz, at 50 Hz
    /// let ts = 1e-3;
    /// let g = TransferFunction::continuous(&[1000.0], &[1.0, 20.0, 1000.0]);
    /// let omega = 2.0 * std::f64::consts::PI * 50.0;
    /// let value = g.discretize(Zoh, ts).unwrap().frequency_transfer_function(ts).response(omega);
    /// let measured = FrequencyResponses(vec![FrequencyResponse { omega, value }]);
    /// let compensated = measured.compensate_zero_order_hold(ts);
    /// let truth = g.frequency_transfer_function().response(omega);
    /// assert!((measured[0].value / truth - 1.0).norm() > 0.1);
    /// assert!((compensated[0].value / truth - 1.0).norm() < 1e-3);
    /// ```
    pub fn compensate_zero_order_hold(&self, ts: T) -> Self {
        self.0.iter().map(|r| FrequencyResponse { omega: r.omega, value: r.value / zero_order_hold_response(r.omega, ts) }).collect()
    }

    /// The rational part `R(jω)` of the responses of a model `G(s) = e^(-delay s) R(s) / s^q`
    /// (`q = integrators`, `delay` \[s\], both known), to fit by a rational function `R = B / A`
    /// (`levy`, `vector_fitting` in
    /// [`system_identification::frequency_response`](crate::system_identification::frequency_response)):
    ///
    /// ```text
    /// R(jω) = G(jω) (jω)^q e^(jω delay)
    /// ```
    ///
    /// Then [`TransferFunctionWithDelay::from_rational_part`](crate::TransferFunctionWithDelay::from_rational_part)
    /// with the same `integrators` and `delay` gives the model `G` from the fitted `R`. The
    /// integrators are poles fixed at the origin, not left to the fit (which would place them near
    /// it), and the delay is not made up for by extra poles. For a measured response of a plant
    /// behind a zero-order hold, apply [`compensate_zero_order_hold`](Self::compensate_zero_order_hold)
    /// first.
    ///
    /// Weights: `|R| = |G| ω^q`, so an absolute-error fit of `R` weights the error of `G` by `ω^q`
    /// (the high frequencies more). To keep the error of `G`, divide the weights `w` of a weighted
    /// fit (`levy::sanathanan_koerner_weighted`, `vector_fitting::identify_weighted`) by `ω^q`,
    /// i.e. fit with `w / ω^q` (`1 / ω^q` for an unweighted one). The relative-error fits need no
    /// change.
    ///
    /// Errors: `ZeroFrequencyWithIntegrators` (`q > 0` and a response at `ω = 0`; leave it out,
    /// e.g. by [`FrequencyResponsesWithCoherence::coherent_bins`]), `InvalidDelay`.
    ///
    /// ```
    /// use dsmc::system_identification::frequency_response::levy;
    /// use dsmc::{FrequencyResponse, FrequencyResponses, TransferFunction, TransferFunctionWithDelay};
    ///
    /// // G(s) = e^(-0.002 s) 100 / (s (s + 10))
    /// let g = TransferFunctionWithDelay::new(TransferFunction::continuous(&[100.0], &[1.0, 10.0, 0.0]), 0.002);
    /// let responses: FrequencyResponses<f64> = (1..=50).map(|k| {
    ///     let omega = 2.0 * k as f64;
    ///     FrequencyResponse { omega, value: g.frequency_response(omega) }
    /// }).collect();
    ///
    /// let rational = responses.extract_rational_part(1, 0.002).unwrap();
    /// let r = levy::identify(&rational, 1, 0).unwrap();                       // 100 / (s + 10)
    /// let model = TransferFunctionWithDelay::from_rational_part(r, 1, 0.002); // e^(-0.002 s) 100 / (s^2 + 10 s)
    /// assert_eq!(model.delay, 0.002);
    /// assert_eq!(model.tf.denominator.len(), 3);
    /// assert!((model.tf.denominator[1] - 10.0).abs() < 1e-8 && model.tf.denominator[2] == 0.0);
    /// ```
    pub fn extract_rational_part(&self, integrators: usize, delay: T) -> Result<Self, RationalPartError> {
        if !delay.is_finite() {
            return Err(RationalPartError::InvalidDelay);
        }
        if integrators > 0
            && let Some(index) = self.0.iter().position(|r| r.omega == T::zero())
        {
            return Err(RationalPartError::ZeroFrequencyWithIntegrators { index });
        }
        Ok(self
            .0
            .iter()
            .map(|r| {
                let factor = Complex::new(T::zero(), r.omega).powi(integrators as i32) * Complex::new(T::zero(), r.omega * delay).exp();
                FrequencyResponse { omega: r.omega, value: r.value * factor }
            })
            .collect())
    }
}

/// Errors of [`FrequencyResponses::extract_rational_part`].
#[derive(Clone, Debug, Error, PartialEq)]
#[non_exhaustive]
pub enum RationalPartError {
    /// A response at `ω = 0` with integrators: `(jω)^q = 0` there, so the rational part is 0
    /// whatever the response (a relative-error fit rejects it, an absolute one is pulled to 0).
    #[error("response {index} at zero frequency with integrators")]
    ZeroFrequencyWithIntegrators { index: usize },
    /// The delay is not finite.
    #[error("delay is not finite")]
    InvalidDelay,
}

impl<T> Deref for FrequencyResponses<T> {
    type Target = [FrequencyResponse<T>];
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<T> IntoIterator for FrequencyResponses<T> {
    type Item = FrequencyResponse<T>;
    type IntoIter = std::vec::IntoIter<FrequencyResponse<T>>;
    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

impl<'a, T> IntoIterator for &'a FrequencyResponses<T> {
    type Item = &'a FrequencyResponse<T>;
    type IntoIter = std::slice::Iter<'a, FrequencyResponse<T>>;
    fn into_iter(self) -> Self::IntoIter {
        self.0.iter()
    }
}

impl<T> FromIterator<FrequencyResponse<T>> for FrequencyResponses<T> {
    fn from_iter<I: IntoIterator<Item = FrequencyResponse<T>>>(iter: I) -> Self {
        Self(iter.into_iter().collect())
    }
}

impl<T> From<Vec<FrequencyResponse<T>>> for FrequencyResponses<T> {
    fn from(val: Vec<FrequencyResponse<T>>) -> Self {
        Self(val)
    }
}

impl<T> From<FrequencyResponses<T>> for Vec<FrequencyResponse<T>> {
    fn from(val: FrequencyResponses<T>) -> Self {
        val.0
    }
}

/// All the responses (with the DC bin, if any), without the coherence.
impl<T> From<FrequencyResponsesWithCoherence<T>> for FrequencyResponses<T> {
    fn from(val: FrequencyResponsesWithCoherence<T>) -> Self {
        val.0.into_iter().map(FrequencyResponse::from).collect()
    }
}

/// Frequency responses with their coherence, one per frequency in order (e.g. the bins of
/// [`welch`](fft::welch)). Derefs to the slice of the responses (`len`, `iter`, indexing), and
/// converts into `Vec<FrequencyResponse<T>>` or [`FrequencyResponses`] (all of them) with `into`.
/// Serialized as the sequence.
#[derive(Clone, Debug, Serialize)]
#[serde(transparent)]
pub struct FrequencyResponsesWithCoherence<T>(pub Vec<FrequencyResponseWithCoherence<T>>);

impl<T: Float> FrequencyResponsesWithCoherence<T> {
    /// The responses that the data define: `ω > 0` and coherence at least `min_coherence` (e.g.
    /// `0.9`), in order, without the coherence. The DC bin is dropped whatever its coherence (that
    /// of [`welch`](fft::welch) is not the static gain: the mean of each segment is removed), and
    /// so are the unexcited bins (coherence zero). The result can be passed to the identification
    /// methods in
    /// [`system_identification::frequency_response`](crate::system_identification::frequency_response)
    /// (as a slice, by `Deref`), and compensated for a zero-order hold
    /// ([`FrequencyResponses::compensate_zero_order_hold`]; the same as compensating first).
    ///
    /// The coherence measures the noise and the nonlinearity, not the excitation: without noise,
    /// the bins between the excited ones (window leakage) also have a coherence close to 1.
    pub fn coherent_bins(&self, min_coherence: T) -> FrequencyResponses<T> {
        self.0
            .iter()
            .filter(|b| b.omega > T::zero() && b.coherence >= min_coherence)
            .map(|&b| b.into())
            .collect()
    }

    /// [`FrequencyResponses::compensate_zero_order_hold`] of the responses; the coherence is kept
    /// (dividing the output by a known factor does not change it), e.g.
    /// `welch(&u, &y, ts, 4).compensate_zero_order_hold(ts).coherent_bins(0.9)`.
    pub fn compensate_zero_order_hold(&self, ts: T) -> Self {
        self.0
            .iter()
            .map(|b| FrequencyResponseWithCoherence { omega: b.omega, value: b.value / zero_order_hold_response(b.omega, ts), coherence: b.coherence })
            .collect()
    }
}

impl<T> Deref for FrequencyResponsesWithCoherence<T> {
    type Target = [FrequencyResponseWithCoherence<T>];
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<T> IntoIterator for FrequencyResponsesWithCoherence<T> {
    type Item = FrequencyResponseWithCoherence<T>;
    type IntoIter = std::vec::IntoIter<FrequencyResponseWithCoherence<T>>;
    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

impl<'a, T> IntoIterator for &'a FrequencyResponsesWithCoherence<T> {
    type Item = &'a FrequencyResponseWithCoherence<T>;
    type IntoIter = std::slice::Iter<'a, FrequencyResponseWithCoherence<T>>;
    fn into_iter(self) -> Self::IntoIter {
        self.0.iter()
    }
}

impl<T> FromIterator<FrequencyResponseWithCoherence<T>> for FrequencyResponsesWithCoherence<T> {
    fn from_iter<I: IntoIterator<Item = FrequencyResponseWithCoherence<T>>>(iter: I) -> Self {
        Self(iter.into_iter().collect())
    }
}

impl<T> From<Vec<FrequencyResponseWithCoherence<T>>> for FrequencyResponsesWithCoherence<T> {
    fn from(val: Vec<FrequencyResponseWithCoherence<T>>) -> Self {
        Self(val)
    }
}

/// All the responses (with the DC bin, if any), without the coherence.
impl<T> From<FrequencyResponsesWithCoherence<T>> for Vec<FrequencyResponse<T>> {
    fn from(val: FrequencyResponsesWithCoherence<T>) -> Self {
        val.0.into_iter().map(FrequencyResponse::from).collect()
    }
}

/// Gain and phase \[rad\] at `frequency`, whose unit is `U` (`Hz` or `RadPerSec`).
#[derive(Copy, Clone, Serialize)]
pub struct FrequencyCharacteristics<T, U = Hz>{
    /// Frequency, in `U`.
    pub frequency: T,
    /// Gain, in dB or as the magnitude depending on how it was computed (`log_scale`).
    pub gain: T,
    /// Phase \[rad\], continuous along the frequency axis from `BodeDiagramPlotter` by default (see `BodeDiagramPlotter::unwrap_phase`), otherwise in `(-π, π]`.
    pub phase: T,
    #[serde(skip)]
    _unit: PhantomData<U>,
}

impl<T: Float, U> FrequencyCharacteristics<T, U> {
    /// All zeros.
    pub fn new() -> Self {
        Self { frequency: T::zero(), gain: T::zero(), phase: T::zero(), _unit: PhantomData }
    }
}

impl<T: Float + FloatConst> FrequencyCharacteristics<T, Hz> {
    /// Convert each response as `From<FrequencyResponse>` does (gain in dB, phase in `(-π, π]`, not unwrapped), with the frequency in Hz.
    pub fn from_responses(responses: &[FrequencyResponse<T>]) -> Vec<Self> {
        responses.iter().map(|&r| r.into()).collect()
    }
}

impl<T: Float + FloatConst> FrequencyCharacteristics<T, RadPerSec> {
    /// `from_responses` with the frequency in rad/s.
    pub fn from_responses_rad_per_sec(responses: &[FrequencyResponse<T>]) -> Vec<Self> {
        responses.iter().map(|&r| r.into()).collect()
    }
}

impl<T: Float, U> Default for FrequencyCharacteristics<T, U> {
    fn default() -> Self {
        Self::new()
    }
}

/// Gain in dB and phase in `(-π, π]`; `omega` [rad/s] is converted into `U`.
impl<T: Float + FloatConst, U: FrequencyUnit> From<FrequencyResponse<T>> for FrequencyCharacteristics<T, U> {
    fn from(response: FrequencyResponse<T>) -> Self {
        Self {
            frequency: U::from_rad_per_sec(response.omega),
            gain: T::from(20.0).unwrap() * response.value.norm().log10(),
            phase: response.value.arg(),
            _unit: PhantomData,
        }
    }
}
