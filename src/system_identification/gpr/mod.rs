use std::ops::{AddAssign, MulAssign};

use num_traits::Float;

pub struct GaussianProcessRegression<T> {
    pub x_sample: Vec<T>,
    pub y_sample: Vec<T>,
    pub x_max: T,
    pub x_min: T,
    kernel: fn(T, T) -> T,
    sense_variance: T,
    inv_cov: Vec<Vec<T>>,
    sample: usize,
}

#[derive(Debug)]
pub struct PredictedValue<T> {
    pub mean: T,
    pub stdev: T,
}

impl<T: Float + AddAssign + MulAssign> GaussianProcessRegression<T> {
    pub fn new(kernel: fn(T, T) -> T, sigma: T) -> Self {
        Self {
            x_sample: vec![],
            y_sample: vec![],
            x_max: T::zero(),
            x_min: T::zero(),
            kernel,
            sense_variance: sigma,
            inv_cov: vec![vec![]],
            sample: 0,
        }
    }

    pub fn add(&mut self, x: T, y: T) {
        self.x_sample.push(x);
        self.y_sample.push(y);

        if self.sample == 0 {
            self.x_max = x;
            self.x_min = x;
        } else {
            if self.x_max < x {
                self.x_max = x;
            }
            if self.x_min > x {
                self.x_min = x;
            }
        }

        self.sample += 1;
    }

    pub fn predict(&mut self, x: T) -> PredictedValue<T> {
        if self.inv_cov.len() != self.sample {
            let mut buffer: Vec<Vec<T>> = vec![vec![T::zero(); self.sample]; self.sample];
            for i in 0..self.sample {
                for j in 0..self.sample {
                    buffer[i][j] = (self.kernel)(self.x_sample[i], self.x_sample[j]);
                }
            }
            self.inv_cov = inverse(&buffer).unwrap();
        }

        let mut k: Vec<T> = vec![T::zero(); self.sample];
        for i in 0..self.sample {
            k[i] = (self.kernel)(self.x_sample[i], x);
        }

        let mut buffer1: Vec<T> = vec![T::zero(); self.sample];
        let mut buffer2: T = T::zero();

        let mut mean: T = T::zero();
        for i in 0..self.sample {
            for j in 0..self.sample {
                buffer1[i] += self.inv_cov[i][j] * self.y_sample[j];
            }
            mean += k[i] * buffer1[i];
        }

        for i in 0..self.sample {
            buffer1[i] = T::zero();
        }

        for i in 0..self.sample {
            for j in 0..self.sample {
                buffer1[i] += self.inv_cov[i][j] * k[j];
            }
            buffer2 += k[i] * buffer1[i];
        }

        let stdev: T = ((self.kernel)(x, x) - buffer2 + self.sense_variance)
            .abs()
            .sqrt();

        PredictedValue { mean, stdev }
    }
}

fn inverse<T: Float + MulAssign>(m: &Vec<Vec<T>>) -> Option<Vec<Vec<T>>> {
    let vlen: usize = m.len();

    let mut m1: Vec<Vec<T>> = m.clone();
    let mut m2: Vec<Vec<T>> = vec![vec![T::zero(); vlen]; vlen];
    for i in 0..vlen {
        m2[i][i] = T::one();
    }

    for i in 0..vlen {
        let mut max_row_option: usize = 0;
        let mut max_value_option: T = T::zero();
        for j in i..vlen {
            if m1[j][i].abs() > max_value_option {
                max_value_option = m1[j][i].abs();
                max_row_option = j;
            }
        }

        /* Error Handling */
        if max_value_option == T::zero() {
            return None;
        }

        let m1_buffer: Vec<T> = m1[i].clone();
        let m2_buffer: Vec<T> = m2[i].clone();
        m1[i] = m1[max_row_option].clone();
        m2[i] = m2[max_row_option].clone();
        m1[max_row_option] = m1_buffer;
        m2[max_row_option] = m2_buffer;

        let scaler: T = T::one() / m1[i][i];
        for j in 0..vlen {
            m1[i][j] *= scaler;
            m2[i][j] *= scaler;
        }

        for j in 0..vlen {
            if i != j {
                let scaler: T = m1[j][i];
                for k in 0..vlen {
                    m1[j][k] = m1[j][k] - scaler * m1[i][k];
                    m2[j][k] = m2[j][k] - scaler * m2[i][k];
                }
            }
        }
    }

    Some(m2)
}
