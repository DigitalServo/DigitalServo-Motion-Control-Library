#[test]
fn test_tf_macro() {
    use dsmc::tf;
    let w = 2.0;
    let tf = tf!("{w}^2 /(2s (s^2 + {w}^2))");
    println!("{}", tf);
}

#[test]
fn test_step_response() {
    use dsmc::tf;
    let tf = tf!("0.3^7/(s+ 0.3)^7");
    let tf_step = tf!("1 / s");
    let tf = tf * tf_step;

    let x = tf.partial_fraction();
    let y = x.time_response(0.0);
    println!("{}", y);
}

#[test]
fn test_closed_tf() {
    use dsmc::tf;
    let tf_open: dsmc::TransferFunction<f64> = tf!("(-2s + 10) / (s^2 + s + 10)");
    let tf_closed = tf_open.unity_feedback();
    println!("{}", tf_closed);
    println!("{}", tf_closed.pz_map());

    // L / (1 + L) = (-2s + 10) / (s^2 - s + 20)
    let expected = tf!("(-2s + 10) / (s^2 - s + 20)");
    assert_eq!(tf_closed.numerator, expected.numerator);
    assert_eq!(tf_closed.denominator, expected.denominator);

    // Same result via `+` / `/` (up to the overall scale of numerator and denominator)
    let via_ops = &tf_open / &(1.0 + &tf_open);
    let scale = via_ops.denominator[0];
    for (a, b) in via_ops.numerator.iter().zip(expected.numerator.iter()) {
        assert!((a / scale - b).abs() < 1e-9);
    }
    for (a, b) in via_ops.denominator.iter().zip(expected.denominator.iter()) {
        assert!((a / scale - b).abs() < 1e-9);
    }
}
