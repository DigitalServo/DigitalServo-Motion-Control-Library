#[test]
fn test_delayer() {
    use dsmc::signal::Delayer;
    let mut delayer = Delayer::new(5);
    for i in 0..20 {
        let out = delayer.output(i);
        // Zero (the default) until the buffer fills, then the input from 5 samples earlier.
        let expected = if i < 5 { 0 } else { i - 5 };
        assert_eq!(out, expected, "sample {i}");
    }
}
