use std::borrow::Borrow;

use num_traits::{float::FloatConst, Float, NumAssignOps, One, ToPrimitive};
use num_complex::Complex;

use crate::TransferFunction;

pub struct NyquistPlotter<T> {
    freq_from: T,
    dfreq: T,
    data_len: usize,
}

impl <T> NyquistPlotter<T>
where
    T: Float + Default + Copy + FloatConst + NumAssignOps + ToPrimitive,
{
    pub fn new(freq_from: T, freq_to: T, dfreq: T) -> Self {
        Self {
            freq_from,
            dfreq,
            data_len: ((freq_to - freq_from) / dfreq).ceil().to_usize().unwrap_or(0)
        }
    }

    /// Calculate the frequency response of a polynomial in the s-domain.
    /// If polynomial is an*s^n +an-1*s^(n-1) + ... + a0, coefficient should be set \[an, an-1, ..., a0\]
    pub fn plot<S: Borrow<TransferFunction<T>>>(&self, tf: S) -> Vec<Complex<T>> {

        let tf = tf.borrow();

        let numer_size = tf.numerator.len();
        let denom_size = tf.denominator.len();

        let mut s: Complex<T>;
        let mut sv: Complex<T>;

        let mut freq: T = self.freq_from;
        let conv_f_to_omega: T = T::from(2.0).unwrap() * FloatConst::PI();

        let mut numer: Complex<T>;
        let mut denom: Complex<T>;

        let mut data: Vec<Complex<T>> = Vec::with_capacity(self.data_len);

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

            let freq_res = numer / denom;

            data.push(freq_res);

            freq += self.dfreq;
        }

        data

    }
}
