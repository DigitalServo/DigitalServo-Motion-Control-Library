use num_traits::{PrimInt, Unsigned};

/// Return n!
pub fn factorial<T: PrimInt + Unsigned>(n: T) -> T {
    if n == T::zero() || n == T::one() {
        return T::one();
    }

    let mut result = T::one();
    let mut i = T::from(2u8).unwrap();

    while i <= n {
        result = result * i;
        i = i + T::one();
    }
    result
}
