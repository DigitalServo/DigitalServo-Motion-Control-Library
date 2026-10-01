use std::marker::PhantomData;

use num_traits::{float::FloatConst, Float, ToPrimitive};

use super::{FrequencyResponse, FrequencyTransferFunction, FrequencyUnit, Hz};

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
    pub fn plot<G: Into<FrequencyTransferFunction<T>>>(&self, g: G) -> Vec<FrequencyResponse<T>> {
        let g = g.into();
        (0..self.data_len)
            .map(|i| {
                let omega = U::to_rad_per_sec(self.freq_from + T::from(i).unwrap() * self.dfreq);
                FrequencyResponse { omega, value: g.response(omega) }
            })
            .collect()
    }
}
