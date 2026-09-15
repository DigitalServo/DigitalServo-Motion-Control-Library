use num_traits::{PrimInt, Unsigned};

/// n Choose k
pub fn binomial_coefficient<T: PrimInt + Unsigned>(n: T, k: T) -> T {
    if k > n {
        return T::zero();
    }

    let k = if k > n - k { n - k } else { k };

    let mut res = T::one();
    let mut i = T::zero();

    while i < k {
        res = res * (n - i);
        res = res / (i + T::one());
        i = i + T::one();
    }
    res
}

/// Returns [C(n,0), ..., C(n,n)]
pub fn binomial_coefficients<T: PrimInt + Unsigned>(n: T) -> Vec<T> {
    if n == T::zero() {
        return vec![T::one()];
    }

    let size = n.to_usize().unwrap_or(0) + 1;
    let mut ret = vec![T::zero(); size];
    ret[0] = T::one();

    let mut current = T::one();
    let n_usize = n.to_usize().unwrap_or(0);

    for k in 1..=n_usize {
        current = current * T::from(n_usize - k + 1).unwrap()
                / T::from(k).unwrap();
        ret[k] = current;
    }
    ret
}
