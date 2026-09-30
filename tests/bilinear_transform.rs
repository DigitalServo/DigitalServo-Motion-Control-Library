use dsmc::{
    BodeDiagramPlotter, TransferFunction, discretize::bilinear_transform::{DiscretizedSystem, discretize}, logger::DataStorage, tf
};

#[test]
fn test_bilinear_transform() {

    let ts = 1e-4;

    let g = 2.0 * std::f64::consts::PI * 10.0;

    let tf_s = TransferFunction::continuous(&[g * g],  &[1.0, 0.01 * g, g * g]);
    let tf_z = discretize(&tf_s, ts);

    let bode_plotter = BodeDiagramPlotter::<f64>::new(0.0, 1000.0, 0.01, true);

    let mut storage_s = DataStorage::new("./out/bilinear_bode_s.csv", ',', false).unwrap();
    let mut storage_z = DataStorage::new("./out/bilinear_bode_z.csv", ',', false).unwrap();

    for res in bode_plotter.frequency_response_s(&tf_s) {
        storage_s.add(&res).unwrap();
    };
    for res in bode_plotter.frequency_response_z(&tf_z, ts) {
        storage_z.add(&res).unwrap();
    };
}

#[test]
fn test_bilinear_transform_filter() {

    let ts = 1e-4;
    let mut storage = DataStorage::new("./out/filter_out.csv", ',', false).unwrap();

    let g = 10.0;

    let tf = TransferFunction::continuous(&[g * g], &[1.0, 2.0 * g, g * g]);

    let mut filter = DiscretizedSystem::new(&tf, ts);

    let mut t = 0.0;
    for _ in 0..20000 {
        let x = if t < 0.2 { 0.0 } else { 1.0 };
        let y = filter.update(x);
        storage.add(&[t, x, y]).unwrap();

        t += ts;
    }

    storage.close().unwrap();
}


#[test]
fn test_discretize_filter() {
    // `f64` fixes `T` for `tf!` below, which cannot infer it before `.abs()` is called.
    let ts: f64 = 1e-4;

    let g = 10.0;
    let tf = tf!("{g} / (s + {g})");
    let tf_d = discretize(&tf, ts);

    // s = a(z - 1)/(z + 1), a = 2/ts:
    // g / (s + g) = g(z + 1) / ((a + g)z + (g - a)), normalized so the leading denominator is 1.
    let a = 2.0 / ts;
    let b = g / (a + g);
    let expected_numer = [b, b];
    let expected_denom = [1.0, (g - a) / (a + g)];
    for (x, e) in tf_d.numerator.iter().zip(expected_numer) {
        assert!((x - e).abs() < 1e-12, "numerator {:?} != {:?}", tf_d.numerator, expected_numer);
    }
    for (x, e) in tf_d.denominator.iter().zip(expected_denom) {
        assert!((x - e).abs() < 1e-12, "denominator {:?} != {:?}", tf_d.denominator, expected_denom);
    }
    assert_eq!(tf_d.numerator.len(), 2);
    assert_eq!(tf_d.denominator.len(), 2);
}
