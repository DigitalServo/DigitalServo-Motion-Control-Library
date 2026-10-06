use dsmc::discretize::Tustin;
use dsmc::{BodeDiagramPlotter, FrequencyTransferFunction, Hz, NyquistPlotter, RadPerSec, tf};
use num_complex::Complex;

#[test]
fn test_frequency_transfer_function() {
    let g: f64 = 2.0 * std::f64::consts::PI * 10.0;
    let k1 = 0.01 * g;
    let k2 = g * g;
    let tf_s = tf!("{k2} / (s^2 + {k1}s + {k2})");
    let g_s: FrequencyTransferFunction<f64> = tf_s.frequency_transfer_function();

    // DC gain 0 dB, -90 deg at the natural frequency
    let c0 = g_s.characteristics::<RadPerSec>(0.0, true);
    assert!(c0.gain.abs() < 1e-9 && c0.phase.abs() < 1e-9);
    let cn = g_s.characteristics::<Hz>(10.0, true);
    assert!((cn.phase + std::f64::consts::FRAC_PI_2).abs() < 1e-9);
    assert_eq!(cn.frequency, 10.0);

    // Linear gain at the natural frequency: k2 / (k1 g)
    let cl = g_s.characteristics::<RadPerSec>(g, false);
    assert!((cl.gain - k2 / (k1 * g)).abs() < 1e-9);
    assert!((cl.phase - cn.phase).abs() < 1e-12);

    // Hz and rad/s plotters agree
    let bode_hz = BodeDiagramPlotter::<f64>::new(1.0, 100.0, 1.0, true).plot(&g_s);
    let bode_rad = BodeDiagramPlotter::<f64, RadPerSec>::new(2.0 * std::f64::consts::PI, 200.0 * std::f64::consts::PI, 2.0 * std::f64::consts::PI, true).plot(&g_s);
    for (a, b) in bode_hz.iter().zip(&bode_rad) {
        assert!((a.gain - b.gain).abs() < 1e-9);
    }

    // Same result as the existing plotter paths
    let bode = BodeDiagramPlotter::<f64>::new(0.1, 100.0, 0.1, true);
    let a = bode.frequency_response_s(&tf_s);
    let b = bode.plot(&g_s);
    let c = bode.plot(&tf_s);
    for ((a, b), c) in a.iter().zip(&b).zip(&c) {
        assert_eq!(a.gain, b.gain);
        assert_eq!(a.gain, c.gain);
    }

    // Discrete: matches z = e^{jωTs}
    let ts = 1e-4;
    let tf_z = tf_s.clone().discretize(Tustin, ts).unwrap();
    let g_z = tf_z.frequency_transfer_function(ts);
    assert!((g_z.response(1.0) - g_s.response(1.0)).norm() < 1e-6);

    // Series connection with a dead time
    let l = 1e-3;
    let delay = FrequencyTransferFunction::new(move |omega: f64| Complex::new(0.0, -omega * l).exp());
    let g_d = &g_s * &delay;
    let w = 50.0;
    assert!((g_d.response(w).norm() - g_s.response(w).norm()).abs() < 1e-12);
    assert!((g_d.characteristics::<RadPerSec>(w, true).phase - (g_s.characteristics::<RadPerSec>(w, true).phase - w * l)).abs() < 1e-12);

    let nyq = NyquistPlotter::<f64>::new(0.1, 100.0, 0.1).plot(&g_d);
    assert_eq!(nyq.len(), b.len());
}
