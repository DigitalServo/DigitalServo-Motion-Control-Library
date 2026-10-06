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

/// Shift each phase by a multiple of 2π so that its step from the previous (unwrapped) phase lies in `(-π, π]`.
fn unwrap_phase<T: Float + FloatConst, U>(points: &mut [FrequencyCharacteristics<T, U>]) {
    let two_pi = T::TAU();
    for i in 1..points.len() {
        let step = points[i].phase - points[i - 1].phase;
        let turns = ((step + T::PI()) / two_pi).ceil() - T::one();
        points[i].phase = points[i].phase - turns * two_pi;
    }
}
