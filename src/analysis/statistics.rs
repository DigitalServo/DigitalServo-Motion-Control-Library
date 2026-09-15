use num_traits;

#[derive(Debug, Copy, Clone)]
pub struct Statistics<T> {
    len: usize,
    offset: T,
    sum: T,
    sum_of_square: T,
    pub mean: T,
    pub variance: T,
    pub sigma: T,
    pub max: T,
    pub min: T,
    pub range: T,
}

impl<T> Statistics<T>
where
    T: num_traits::Float + std::ops::AddAssign,
{
    pub fn new(offset: T) -> Self {
        Self {
            offset,
            len: 0,
            sum: T::zero(),
            sum_of_square: T::zero(),
            mean: T::zero(),
            variance: T::zero(),
            sigma: T::zero(),
            max: T::zero(),
            min: T::zero(),
            range: T::zero(),
        }
    }

    pub fn add(&mut self, data: T) {
        self.len += 1;

        let data_trim: T = data - self.offset;
        self.sum += data_trim;
        self.sum_of_square += data_trim.powi(2);

        let len_t: T = T::from(self.len).unwrap();

        let mu_trim: T = self.sum / len_t;
        let var_trim: T = self.sum_of_square / len_t;

        self.mean = mu_trim + self.offset;
        self.variance = var_trim - mu_trim.powi(2);
        self.sigma = self.variance.sqrt();

        if self.len == 1 {
            self.max = data;
            self.min = data;
        } else {
            if self.max < data {
                self.max = data
            };
            if self.min > data {
                self.min = data
            };
        }

        self.range = self.max - self.min;
    }
}
