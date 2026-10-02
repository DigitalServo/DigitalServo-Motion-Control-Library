use std::borrow::Borrow;
use std::marker::PhantomData;

use num_traits::{float::FloatConst, Float, ToPrimitive};

use crate::{Continuous, Discrete, TransferFunction};

use super::{FrequencyCharacteristics, FrequencyTransferFunction, FrequencyUnit, Hz};

/// Bode diagram over `[freq_from, freq_to)` with step `dfreq`, all given in `U` (`Hz` by default).
pub struct BodeDiagramPlotter<T, U = Hz> {
    freq_from: T,
    dfreq: T,
    data_len: usize,
    log_scale: bool,
    _unit: PhantomData<U>,
}

impl <T, U> BodeDiagramPlotter<T, U>
where
    T: Float + FloatConst + ToPrimitive + 'static,
    U: FrequencyUnit,
{
    /// `data_len = ceil((freq_to - freq_from) / dfreq)` points starting at `freq_from`, in `U`.
    /// Gain in dB if `log_scale`, otherwise the magnitude.
    pub fn new(freq_from: T, freq_to: T, dfreq: T, log_scale: bool) -> Self {
        Self {
            freq_from,
            dfreq,
            data_len: ((freq_to - freq_from) / dfreq).ceil().to_usize().unwrap_or(0),
            log_scale,
            _unit: PhantomData,
        }
    }

    /// Calculate the frequency characteristics of a frequency transfer function
    /// (or anything convertible into one, such as a continuous `TransferFunction`).
    pub fn plot<G: Into<FrequencyTransferFunction<T>>>(&self, g: G) -> Vec<FrequencyCharacteristics<T, U>> {
        let g = g.into();
        (0..self.data_len)
            .map(|i| g.characteristics(self.freq_from + T::from(i).unwrap() * self.dfreq, self.log_scale))
            .collect()
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
