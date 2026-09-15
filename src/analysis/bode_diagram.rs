use std::borrow::Borrow;

use num_traits::{float::FloatConst, Float, NumAssignOps, One, ToPrimitive};
use num_complex::Complex;

use crate::TransferFunction;

use super::FrequencyCharacteristics;

pub struct BodeDiagramPlotter<T> {
    freq_from: T,
    dfreq: T,
    data_len: usize,
    log_scale: bool,
}

impl <T> BodeDiagramPlotter<T>
where
    T: Float + Default + Copy + FloatConst + NumAssignOps + ToPrimitive,
{
    pub fn new(freq_from: T, freq_to: T, dfreq: T, log_scale: bool) -> Self {
        Self {
            freq_from,
            dfreq,
            data_len: ((freq_to - freq_from) / dfreq).ceil().to_usize().unwrap_or(0),
            log_scale,
        }
    }

    /// Calculate the frequency response of a polynomial in the s-domain.
    /// If polynomial is an*s^n +an-1*s^(n-1) + ... + a0, coefficient should be set \[an, an-1, ..., a0\]
    pub fn frequency_response_s<S: Borrow<TransferFunction<T>>>(&self, tf: S) -> Vec<FrequencyCharacteristics<T>> {

        let tf = tf.borrow();

        let numer_size = tf.numerator.len();
        let denom_size = tf.denominator.len();

        let mut s: Complex<T>;
        let mut sv: Complex<T>;
        let mut freq_res: Complex<T>;

        let mut freq: T = self.freq_from;
        let conv_f_to_omega: T = T::from(2.0).unwrap() * FloatConst::PI();

        let mut numer: Complex<T>;
        let mut denom: Complex<T>;

        let mut data: Vec<FrequencyCharacteristics<T>> = Vec::with_capacity(self.data_len);

        for _ in 0..self.data_len {
            numer = Complex::default();
            denom = Complex::default();

            let omega: T = conv_f_to_omega * freq;
            s = Complex::new(T::zero(), omega);

            sv = Complex::one();
            for i in 1..=numer_size {
                numer += sv * tf.numerator[numer_size - i];
                sv *= s;
            }

            sv = Complex::one();
            for i in 1..=denom_size {
                denom +=  sv * tf.denominator[denom_size - i];
                sv *= s;
            }

            freq_res = numer / denom;

            let gain: T = if self.log_scale {
                T::from(20.0).unwrap() * freq_res.norm().log10()
            } else {
                freq_res.norm()
            };
            let phase: T = (freq_res.im).atan2(freq_res.re);

            data.push(FrequencyCharacteristics { frequency: freq, gain, phase });

            freq += self.dfreq;
        }

        data

    }

    /// Calculate the frequency response of a polynomial in the z-domain.
    /// If polynomial is an*z^n +an-1*z^(n-1) + ... + a0, coefficient should be set \[an, an-1, ..., a0\]
    pub fn frequency_response_z<S: Borrow<TransferFunction<T>>>(&self, tf: S,  ts: T) -> Vec<FrequencyCharacteristics<T>> {

        let tf = tf.borrow();

        let numer_size = tf.numerator.len();
        let denom_size = tf.denominator.len();

        let mut z: Complex<T>;
        let mut zv: Complex<T>;
        let mut freq_res: Complex<T>;

        let mut freq: T = self.freq_from;
        let conv_f_to_omega: T = T::from(2.0).unwrap() * FloatConst::PI();

        let mut numer: Complex<T>;
        let mut denom: Complex<T>;

        let mut data: Vec<FrequencyCharacteristics<T>> = Vec::with_capacity(self.data_len);

        for _ in 0..self.data_len {
            numer = Complex::default();
            denom = Complex::default();

            let omega: T = conv_f_to_omega * freq;
            z = Complex::new(T::zero(), omega * ts).exp();

            zv = Complex::one();
            for i in 1..=numer_size {
                numer += zv * tf.numerator[numer_size - i];
                zv *= z;
            }

            zv = Complex::one();
            for i in 1..=denom_size {
                denom +=  zv * tf.denominator[denom_size - i];
                zv *= z;
            }

            freq_res = numer / denom;

            let gain: T = if self.log_scale {
                T::from(20.0).unwrap() * freq_res.norm().log10()
            } else {
                freq_res.norm()
            };
            let phase: T = (freq_res.im).atan2(freq_res.re);

            data.push(FrequencyCharacteristics { frequency: freq, gain, phase });

            freq += self.dfreq;
        }

        data

    }
}
