/// Running statistics of a data stream, updated by `add`.
/// Sums are accumulated relative to `offset` (a value close to the data) to reduce cancellation
/// in the variance.
#[derive(Debug, Copy, Clone)]
pub struct Statistics<T> {
    len: usize,
    offset: T,
    sum: T,
    sum_of_square: T,
    /// Mean.
    pub mean: T,
    /// Population variance (divided by the number of samples).
    pub variance: T,
    /// Standard deviation, `sqrt(variance)`.
    pub sigma: T,
    /// Largest value.
    pub max: T,
    /// Smallest value.
    pub min: T,
    /// `max - min`.
    pub range: T,
}

impl<T> Statistics<T>
where
    T: num_traits::Float + std::ops::AddAssign,
{
    /// Empty statistics. `offset` should be close to the data (e.g. the first sample); it only affects round-off.
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

    /// Add a sample and update all statistics.
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
