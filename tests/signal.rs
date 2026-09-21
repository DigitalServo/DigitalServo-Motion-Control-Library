use dsmc::factorial;

#[test]
fn test_delayer() {
    use dsmc::signal::Delayer;
    let mut delayer = Delayer::new(5);
    for i in 0..20 {
        let out = delayer.output(i);
        println!("{:?}, {:?}", i, out);
    }
}

#[test]
fn test_differentiator() {
    use dsmc::logger::DataStorage;
    use dsmc::signal::Differentiator;

    let ts: f64 = 1e-5;
    let mut t: f64 = 0.0;

    let bandwidth = 100.0;

    let derivative_order = 1;
    let filter_order = 0;

    let mut storage = DataStorage::new("./out/differentiator.csv", ',', false).unwrap();
    let mut differentiator = Differentiator::new(ts, bandwidth, derivative_order, filter_order);

    for _ in 0..100000 {
        let input = t.powi(derivative_order as i32) / factorial(derivative_order as u64) as f64;
        let out = differentiator.update(input);
        storage.add(&[t, out]).unwrap();

        t += ts;
    }

}
