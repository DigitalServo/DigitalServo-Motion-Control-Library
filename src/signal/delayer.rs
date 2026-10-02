/// Pure delay of `delay_sample` samples (ring buffer), initially filled with `T::default()`.
pub struct Delayer<T> {
    buffer: Vec<T>,
    index: usize,
    delay_sample: usize
}

impl<T: Sized + Default + Copy> Delayer<T> {
    /// Delay of `delay_sample` samples (at least 1).
    pub fn new(delay_sample: usize) -> Self {
        Self {
            buffer: vec![T::default(); delay_sample],
            index: 0,
            delay_sample
        }
    }

    /// Push `u[k]` and return `u[k - delay_sample]`.
    pub fn output(&mut self, u: T) -> T {
        let out: T = self.buffer[self.index];
        self.buffer[self.index] = u;
        self.index = if self.index < (self.delay_sample - 1) {
            self.index + 1
        } else {
            0
        };

        out
    }

    /// Clear the buffer to `T::default()`.
    pub fn reset(&mut self) {
        self.buffer = vec![T::default(); self.delay_sample];
        self.index = 0;
    }
}
