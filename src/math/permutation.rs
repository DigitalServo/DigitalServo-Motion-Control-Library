use num_traits::{PrimInt, Unsigned};

/// Return nPr = n! / (n-r)!
pub fn permutation<T: PrimInt + Unsigned>(n: T, r: T) -> T {
    if r == T::zero() {
        return T::one();
    }
    if n < r {
        return T::zero();
    }

    let mut result = T::one();
    let mut i = n;
    let stop = n - r;

    while i > stop {
        result = result * i;
        i = i - T::one();
    }
    result
}
