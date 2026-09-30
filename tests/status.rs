use dsmc::status::Motion;


#[test]
fn test_motion_scalar_ops() {
    let m = Motion{x: 1.0, v: -2.0, a: 4.0, f: 0.5};

    // Mul and Div
    assert_eq!(m * 2.0, Motion{x: 2.0, v: -4.0, a: 8.0, f: 1.0});
    assert_eq!(m / 2.0, Motion{x: 0.5, v: -1.0, a: 2.0, f: 0.25});

    // MulAssign and DivAssign
    let mut n = m;
    n *= 4.0;
    assert_eq!(n, Motion{x: 4.0, v: -8.0, a: 16.0, f: 2.0});
    n /= 8.0;
    assert_eq!(n, Motion{x: 0.5, v: -1.0, a: 2.0, f: 0.25});

    // `Motion` is `Copy`, so the original is untouched.
    assert_eq!(m, Motion{x: 1.0, v: -2.0, a: 4.0, f: 0.5});
}
