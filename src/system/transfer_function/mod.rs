use crate::Polynomial;

#[derive(Clone, Debug)]
pub struct TransferFunction<T> {
    pub numerator: Polynomial<T>,
    pub denominator: Polynomial<T>,
}

impl<T: Clone> TransferFunction<T> {
    pub fn new(numerator: &[T], denominator: &[T]) -> Self {
        Self {
            numerator: Polynomial(numerator.to_vec()),
            denominator: Polynomial(denominator.to_vec()),
        }
    }
}
