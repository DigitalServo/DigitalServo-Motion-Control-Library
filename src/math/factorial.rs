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

/// Return nPr = n! / (n-r)!
pub fn factorial_n_to_r<T: PrimInt + Unsigned>(n: T, r: T) -> T {
    if n < r {
        return T::zero();
    }
    if n == r || r == T::zero() {
        return T::one();
    }

    let mut result = n;
    let mut i = n - T::one();

    while i >= r {
        result = result * i;
        i = i - T::one();
    }
    result
}
