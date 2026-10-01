use std::marker::PhantomData;
use std::ops::Mul;
use std::rc::Rc;

use num_complex::Complex;
use num_traits::{Float, FloatConst};

use crate::{Continuous, Discrete, TransferFunction};

use super::{FrequencyCharacteristics, FrequencyUnit};

/// Frequency transfer function `ω [rad/s] -> G(jω)`.
/// Built from a `TransferFunction` (`s = jω` / `z = e^{jωTs}`) or from any closure,
/// so that elements without a rational form (e.g. dead time `e^{-jωL}`) can also be handled.
pub struct FrequencyTransferFunction<T> {
    response: Rc<dyn Fn(T) -> Complex<T>>,
}

impl<T> Clone for FrequencyTransferFunction<T> {
    fn clone(&self) -> Self {
        Self { response: Rc::clone(&self.response) }
    }
}

impl<T: Float + FloatConst + 'static> FrequencyTransferFunction<T> {
    pub fn new(response: impl Fn(T) -> Complex<T> + 'static) -> Self {
        Self { response: Rc::new(response) }
    }

    /// Complex response G(jω).
    pub fn response(&self, omega: T) -> Complex<T> {
        (self.response)(omega)
    }

    /// Gain (in dB if `log_scale`, otherwise the magnitude) and phase \[rad\] at `frequency` given in `U`.
    pub fn characteristics<U: FrequencyUnit>(&self, frequency: T, log_scale: bool) -> FrequencyCharacteristics<T, U> {
        let g = self.response(U::to_rad_per_sec(frequency));
        let gain = if log_scale { T::from(20.0).unwrap() * g.norm().log10() } else { g.norm() };
        FrequencyCharacteristics { frequency, gain, phase: g.arg(), _unit: PhantomData }
    }

    /// `|frequency| -> FrequencyCharacteristics` closure (see `characteristics`).
    pub fn characteristics_fn<U: FrequencyUnit>(&self, log_scale: bool) -> impl Fn(T) -> FrequencyCharacteristics<T, U> + '_ {
        move |frequency| self.characteristics(frequency, log_scale)
    }
}

/// Evaluate a descending-order real polynomial at a complex point (Horner's method).
fn eval_polynomial<T: Float>(coefficients: &[T], x: Complex<T>) -> Complex<T> {
    coefficients.iter().fold(Complex::new(T::zero(), T::zero()), |acc, &c| acc * x + c)
}

impl<T: Float + FloatConst + 'static> TransferFunction<T, Continuous> {
    /// G(jω) with `s = jω`.
    pub fn frequency_transfer_function(&self) -> FrequencyTransferFunction<T> {
        let numer = self.numerator.0.clone();
        let denom = self.denominator.0.clone();
        FrequencyTransferFunction::new(move |omega| {
            let s = Complex::new(T::zero(), omega);
            eval_polynomial(&numer, s) / eval_polynomial(&denom, s)
        })
    }
}

impl<T: Float + FloatConst + 'static> TransferFunction<T, Discrete> {
    /// G(e^{jωTs}) with sampling time `ts`.
    pub fn frequency_transfer_function(&self, ts: T) -> FrequencyTransferFunction<T> {
        let numer = self.numerator.0.clone();
        let denom = self.denominator.0.clone();
        FrequencyTransferFunction::new(move |omega| {
            let z = Complex::new(T::zero(), omega * ts).exp();
            eval_polynomial(&numer, z) / eval_polynomial(&denom, z)
        })
    }
}

impl<T: Float + FloatConst + 'static> From<&TransferFunction<T, Continuous>> for FrequencyTransferFunction<T> {
    fn from(tf: &TransferFunction<T, Continuous>) -> Self {
        tf.frequency_transfer_function()
    }
}

impl<T: Float + FloatConst + 'static> From<TransferFunction<T, Continuous>> for FrequencyTransferFunction<T> {
    fn from(tf: TransferFunction<T, Continuous>) -> Self {
        tf.frequency_transfer_function()
    }
}

impl<T> From<&FrequencyTransferFunction<T>> for FrequencyTransferFunction<T> {
    fn from(g: &FrequencyTransferFunction<T>) -> Self {
        g.clone()
    }
}

/// Series connection G1(jω) G2(jω), e.g. a continuous plant with a discrete controller.
impl<T: Float + FloatConst + 'static> Mul for FrequencyTransferFunction<T> {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        Self::new(move |omega| self.response(omega) * rhs.response(omega))
    }
}

impl<T: Float + FloatConst + 'static> Mul for &FrequencyTransferFunction<T> {
    type Output = FrequencyTransferFunction<T>;
    fn mul(self, rhs: Self) -> FrequencyTransferFunction<T> {
        self.clone() * rhs.clone()
    }
}
