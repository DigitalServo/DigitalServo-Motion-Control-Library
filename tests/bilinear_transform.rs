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

    let ts = 1e-4;

    let tf = tf!("10.0 / (s + 10.0)", 's');
    let tf_d = discretize(&tf, ts);

    println!("{}", tf_d);
}
