#[test]
fn test_lsm_arx() {
    use mclib::{TransferFunction, discretize::bilinear_transform, system_identification::{kalman_filter, lsm}};

    let ts: f64 = 1e-3;

    let tf_c = TransferFunction::new(&[1000.0], &[1.0, 20.0, 1000.0]);

    let mut lsm = lsm::arx::DataBuffer::<f64>::new(2, 2);
    let mut kf = kalman_filter::arx::KalmanFilter::<f64>::new(2, 2, 0.01, 0.01, 1.0e10);

    let mut system = bilinear_transform::DiscretizedSystem::new(&tf_c, ts);

    let mut t = 0.0;
    for _ in 0..1000 {
        let x_prev = system.output;

        let mut u = 0.0;
        for i in 0..20 {
            u += 0.1 * ((i as f64) * t).sin();
        }

        let y = system.update(u);
        lsm.add(u, x_prev, y);
        kf.update(u, x_prev, y);

        t += ts;
    }

    let tf_z = bilinear_transform::discretize(&tf_c, ts);

    println!("Bilinear: {:.6?}", tf_z);
    println!("LS method: {:.6?}", lsm.identify().unwrap());
    println!("Kalman flt: {:.6?}", kf.identify());
}


#[test]
fn test_lsm_polynomial() {
    use mclib::system_identification::{kalman_filter, lsm};

    let dx: f64 = 1e-3;

    let mut lsm = lsm::polynomial::DataBuffer::<f64>::new(3);
    let mut kf = kalman_filter::polynomial::KalmanFilter::<f64>::new(3, 0.0, 0.0 ,1.0e5);

    let generator = |x: f64| -> f64 {
        3.0 * x.powi(3) - 1.0 * x.powi(2) + 0.3 * x.powi(1) + 0.05
    };

    let mut x = 0.0;
    for _ in 0..1000 {
        let y: f64 = generator(x);
        lsm.add(x, y);
        kf.update(x, y);
        x += dx;
    }

    println!("lsm: {:.02?}", lsm.identify());
    println!("kf: {:.02?}", kf.identify());
}


#[test]
fn test_levy() {

    use nalgebra::Complex;
    use mclib::FrequencyResponse;
    use mclib::system_identification::frequency_response::levy;

    fn system(omega: f64) -> Complex<f64> {
        let g = Complex::new(100.0, 0.0);
        let s = Complex::new(0.0, omega);
        let numer = -3.0 * s.powi(2) + 10.0 * s.powi(1) + g;
        let denom = 2.0 * s.powi(3) + 4.0 * s.powi(2) + (2.0 * g) * s.powi(1) + g * g;
        numer / denom
    }

    let size = 100;
    let mut samples: Vec<FrequencyResponse<f64>> = Vec::with_capacity(size);

    let numer_order = 2;
    let denom_order = 3;

    for i in 0..size {
        let omega = 2.0 * std::f64::consts::PI * (i as f64);
        let value = system(omega);
        samples.push(FrequencyResponse { omega, value });
    }

    let ret = levy::sanathanan_koerner_identification(&samples, numer_order, denom_order, 5).unwrap();
    println!("{:.02?}", ret);

}

#[test]
fn test_vector_fitting() {
    use std::f64::consts::PI;
    use mclib::{FrequencyResponse, TransferFunction, system_identification::frequency_response::vector_fitting::{VectorFittingOptions, VectorFittingResult, identify}};
    use num_complex::Complex64;

    // poles and residues of the true function
    let true_poles = [
        Complex64::new(-0.0, 1.0),
        Complex64::new(-0.0, -1.0),
    ];
    let true_residues = [
        Complex64::new(1.0, 0.0),
        Complex64::new(1.0, 0.0),
    ];

    // samples
    // DONOT INCLUDE DC Component freq = 0
    let freqs: Vec<f64> = (1..=1000)
        .map(|k| 0.1 * k as f64)
        .collect();

    let samples: Vec<FrequencyResponse<f64>> = freqs
        .iter()
        .map(|&f| {
            let s = Complex64::new(0.0, 2.0 * PI * f);
            let mut h = Complex64::new(0.0, 0.0);
            for (&a, &c) in true_poles.iter().zip(true_residues.iter()) {
                h += c / (s - a);
            }
            FrequencyResponse { omega: s.im, value: h }
        })
        .collect();

    let mut rms = f64::INFINITY;
    let mut ret: Option<(usize, VectorFittingResult<f64>)> = None;
    for order in 1..=5 {
        let mut opts = VectorFittingOptions::default();
        opts.fit_d = true;
        opts.fit_e = true;
        let result = identify(&samples, order, &opts).unwrap();

        if rms > *result.rms_errors.last().unwrap() {
            rms = *result.rms_errors.last().unwrap();
            ret = Some((order, result))
        }
    }

    let (order, result) = ret.unwrap();

    println!("Order: {order}");
    println!("Final RMS error: {:.2e}", rms);
    println!("Fitted poles: {:.2?}", result.poles);
    println!("Fitted residues: {:.2?}", result.residues);
    assert!(
        rms < 1e-4,
        "RMS error too large: {:.2e}",
        rms
    );

    let tf: TransferFunction<f64> = result.into();
    println!("Transfer function: {:.2?}", tf);

}


#[test]
fn test_identification_from_simulation() {
    use num_complex::Complex;
    use mclib::TransferFunction;
    use mclib::discretize::bilinear_transform;
    use mclib::FrequencyResponse;
    use mclib::fft::{fft, welch};
    use mclib::logger::DataStorage;
    use mclib::system_identification::{kalman_filter,lsm};
    use mclib::BodeDiagramPlotter;

    let iterations = 10000;

    let ts: f64 = 1e-3;

    let f_nyquist = 1.0 / (2.0 * ts);

    let g = 20.0;
    let tf_s = TransferFunction::new(&[g * g], &[1.0, 0.1 * g, g * g]);
    let tf_z = bilinear_transform::discretize(&tf_s, ts);

    let mut system = bilinear_transform::DiscretizedSystem::new(&tf_s, ts);

    let simulator = |omega: f64| -> Complex<f64> {
        let s = Complex::new(0.0, omega);
        let numer = g * g;
        let denom = s.powi(2) + 0.1 * g * s.powi(1) + g * g;
        numer / denom
    };

    let input_order = 2;
    let state_order = 2;
    let mut lsm = lsm::arx::DataBuffer::<f64>::new(input_order, state_order);
    let mut kf = kalman_filter::arx::KalmanFilter::<f64>::new(input_order, state_order, 0.01, 0.01, 1.0e10);

    let mut input: Vec<f64> = Vec::with_capacity(iterations);
    let mut output: Vec<f64> = Vec::with_capacity(iterations);

    let mut t = 0.0;
    for _ in 1..iterations {

        let x_prev = system.output;

        let mut u = 0.0;
        for i in 1..=100 {
            let omega = 2.0 * std::f64::consts::PI * (f_nyquist) * (i as f64) / 300.0;
            u += 1.0 * (omega * t).sin();
        }

        input.push(u);
        output.push(system.output);

        let y = system.update(u);

        lsm.add(u, x_prev, y);
        kf.update(u, x_prev, y);

        t += ts;
    }

    let s1 = {
        let u = fft(&input, ts);
        let y = fft(&output, ts);
        let g: Vec<FrequencyResponse<f64>> = y
            .iter()
            .zip(u.iter())
            .map(|(y, u)| FrequencyResponse{
                omega: y.omega,
                value: y.value / u.value,
            })
            .collect::<Vec<_>>();

        g[5..].to_vec()
    };

    let mut s2: Vec<FrequencyResponse<f64>> = Vec::with_capacity(s1.len());
    for x in &s1 {
        let omega = x.omega;
        let value = simulator(omega);
        s2.push(FrequencyResponse { omega, value });
    }

    let (s3, _coherence) = welch(&input, &output, ts, 20);
    let s3 = s3[1..40].to_vec();

    let mut out = DataStorage::new("./out/si.csv", ',', false).unwrap();
    for (v1, v2) in s1.iter().zip(s2.iter()) {
        // out.add(&[v1.omega, v1.value.re, v1.value.im, v2.value.re, v2.value.im]).unwrap();
        out.add(&[v1.omega, v1.value.norm(), v1.value.im.atan2(v1.value.re), v2.value.norm(), v2.value.im.atan2(v2.value.re)]).unwrap();
    }

    let mut out2 = DataStorage::new("./out/si2.csv", ',', false).unwrap();
    for v in &s3 {
        // out.add(&[v1.omega, v1.value.re, v1.value.im, v2.value.re, v2.value.im]).unwrap();
        out2.add(&[v.omega, v.value.norm(), v.value.im.atan2(v.value.re)]).unwrap();
    }

    let tf_z_lsm = lsm.identify().unwrap();
    let tf_z_kf = kf.identify();

    println!("Simulator: {:.6?}", tf_z);
    println!("LS method: {:.6?}", tf_z_lsm);
    println!("Kalman flt: {:.6?}", tf_z_kf);

    let bode_plotter = BodeDiagramPlotter::<f64>::new(0.0, 50.0,  0.01, false);

    let s4 = bode_plotter.frequency_response_z(&tf_z_lsm, ts);
    let s5 = bode_plotter.frequency_response_z(&tf_z_kf, ts);

    let mut out3 = DataStorage::new("./out/si3.csv", ',', false).unwrap();
    for (&v1, &v2) in s4.iter().zip(s5.iter()) {
        let omega = v1.frequency * 2.0 * std::f64::consts::PI;
        out3.add(&[omega, v1.gain, v1.phase, v2.gain, v2.phase]).unwrap();
    }

    // use mclib::system_identification::frequency_response::vector_fitting::{identify, VectorFittingOptions, VectorFittingResult};

    // let mut rms = f64::INFINITY;
    // let mut ret: Option<(usize, VectorFittingResult<f64>)> = None;

    // let opts = VectorFittingOptions::default();
    // for order in 1..=3 {
    //     let result = identify(&s3, order, &opts).unwrap();
    //     if rms > *result.rms_errors.last().unwrap() {
    //         rms = *result.rms_errors.last().unwrap();
    //         ret = Some((order, result))
    //     }
    // }

    // let (_, result) = ret.unwrap();

    // let tf: TransferFunction<f64> = result.into();
    // println!("Transfer function: {:.2?}", tf);

}
