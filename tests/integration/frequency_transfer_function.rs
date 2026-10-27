use dsmc::discretize::Tustin;
use dsmc::{BodeDiagramPlotter, FrequencyCharacteristics, FrequencyTransferFunction, Hz, NyquistPlotter, RadPerSec, TransferFunctionWithDelay, tf};
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
    assert_eq!(g_s.with_delay(l).response(w), g_d.response(w));
    assert!((g_d.response(w).norm() - g_s.response(w).norm()).abs() < 1e-12);
    assert!((g_d.characteristics::<RadPerSec>(w, true).phase - (g_s.characteristics::<RadPerSec>(w, true).phase - w * l)).abs() < 1e-12);

    let nyq = NyquistPlotter::<f64>::new(0.1, 100.0, 0.1).plot(&g_d);
    assert_eq!(nyq.len(), b.len());

    // FrequencyResponse -> FrequencyCharacteristics: dB gain, wrapped phase, frequency in U
    let bode_dw = BodeDiagramPlotter::<f64>::new(0.1, 100.0, 0.1, true).unwrap_phase(false).plot(&g_d);
    for (r, c) in nyq.iter().zip(&bode_dw) {
        let hz: FrequencyCharacteristics<f64> = (*r).into();
        assert!((hz.frequency - c.frequency).abs() < 1e-12);
        assert!((hz.gain - c.gain).abs() < 1e-12 && (hz.phase - c.phase).abs() < 1e-12);
        let rad = FrequencyCharacteristics::<f64, RadPerSec>::from(*r);
        assert_eq!(rad.frequency, r.omega);
    }
    let from_vec = FrequencyCharacteristics::from_responses(&nyq);
    let from_vec_rad = FrequencyCharacteristics::from_responses_rad_per_sec(&nyq);
    assert_eq!(from_vec.len(), bode_dw.len());
    for (((a, b), c), r) in from_vec.iter().zip(&from_vec_rad).zip(&bode_dw).zip(&nyq) {
        assert!((a.frequency - c.frequency).abs() < 1e-12);
        assert_eq!(b.frequency, r.omega);
        assert!((a.gain - c.gain).abs() < 1e-12 && (a.phase - c.phase).abs() < 1e-12);
        assert!(a.gain == b.gain && a.phase == b.phase);
    }

    // TransferFunctionWithDelay plots like the series connection above
    let tf_d = TransferFunctionWithDelay::new(tf_s.clone(), l);
    let nyq_d = NyquistPlotter::<f64>::new(0.1, 100.0, 0.1).plot(&tf_d);
    for (a, b) in nyq.iter().zip(&nyq_d) {
        assert!((a.value - b.value).norm() < 1e-12);
    }
    let bode_d = bode.plot(tf_d.clone());
    let bode_g = bode.plot(&g_d);
    for (a, b) in bode_d.iter().zip(&bode_g) {
        assert!((a.gain - b.gain).abs() < 1e-12 && (a.phase - b.phase).abs() < 1e-12);
    }

    // Unwrapped phase: phase of tf_s minus omega * l, continuous over many turns
    let bode_u = BodeDiagramPlotter::<f64>::new(0.1, 2000.0, 0.1, true).plot(&tf_d);
    let bode_w = BodeDiagramPlotter::<f64>::new(0.1, 2000.0, 0.1, true).unwrap_phase(false).plot(&tf_d);
    let bode_s = BodeDiagramPlotter::<f64>::new(0.1, 2000.0, 0.1, true).plot(&tf_s);
    for ((u, w), s) in bode_u.iter().zip(&bode_w).zip(&bode_s) {
        let omega = 2.0 * std::f64::consts::PI * u.frequency;
        assert!((u.phase - (s.phase - omega * l)).abs() < 1e-9);
        let turns = (u.phase - w.phase) / (2.0 * std::f64::consts::PI);
        assert!((turns - turns.round()).abs() < 1e-9);
        assert_eq!(u.gain, w.gain);
        assert!(w.phase > -std::f64::consts::PI && w.phase <= std::f64::consts::PI);
    }
    assert!(bode_u.last().unwrap().phase < -4.0 * std::f64::consts::PI);
}

#[test]
fn test_adaptive_nyquist() {
    use dsmc::{AdaptiveNyquistPlotter, TransferFunction};

    // Rigid body with a lightly damped mode at 300 Hz (ζ = 0.001) under a PID controller
    let wr = 2.0 * std::f64::consts::PI * 300.0;
    let zeta = 1e-3;
    let plant = TransferFunction::continuous(&[wr * wr], &[1.0, 0.0, 0.0]) * TransferFunction::continuous(&[1.0], &[1.0, 2.0 * zeta * wr, wr * wr]);
    let omega = 300.0;
    let pid = TransferFunction::continuous(&[3.0 * omega, 3.0 * omega * omega, omega * omega * omega], &[1.0, 0.0]);
    let tf = pid * plant;
    let g = tf.frequency_transfer_function();

    let max_angle = 2f64.to_radians();
    let nyq = AdaptiveNyquistPlotter::<f64>::new(10.0, 2000.0).plot(&tf);
    let fixed = NyquistPlotter::<f64>::new(10.0, 2000.0, 0.01).plot(&tf);

    // Both ends included, strictly increasing, far fewer points than a fixed step of 0.01 Hz
    assert!((nyq[0].omega - 2.0 * std::f64::consts::PI * 10.0).abs() < 1e-9);
    assert!((nyq.last().unwrap().omega - 2.0 * std::f64::consts::PI * 2000.0).abs() < 1e-9);
    assert!(nyq.windows(2).all(|w| w[1].omega > w[0].omega));
    assert!(nyq.len() * 20 < fixed.len(), "{} points", nyq.len());

    // Smooth: the locus turns little at each point and |ln(G_2 / G_1)| is bounded
    for w in nyq.windows(3) {
        let turn = ((w[2].value - w[1].value) / (w[1].value - w[0].value)).arg().abs();
        assert!(turn < 2.0 * max_angle, "turn {turn} at {}", w[1].omega);
    }
    for w in nyq.windows(2) {
        assert!((w[1].value / w[0].value).ln().norm() < 0.1 + 1e-12);
    }

    // The resonance circle is resolved: its peak |G| is reached to within 0.1 %
    let peak = |r: &[dsmc::FrequencyResponse<f64>]| r.iter().map(|r| r.value.norm()).fold(0.0, f64::max);
    let peak_true = (0..=20000).map(|i| g.response(wr * (0.99 + 0.02 * i as f64 / 20000.0)).norm()).fold(0.0, f64::max);
    assert!((peak(&nyq) / peak_true - 1.0).abs() < 1e-3);

    // From zero frequency: a first-order lag ends at G(0) = 1
    let lag = AdaptiveNyquistPlotter::<f64>::new(0.0, 100.0).plot(tf!("10 / (s + 10)"));
    assert_eq!(lag[0].omega, 0.0);
    assert!((lag[0].value.re - 1.0).abs() < 1e-12);

    // The bound on the number of points
    let capped = AdaptiveNyquistPlotter::<f64>::new(10.0, 2000.0).max_points(100).plot(&tf);
    assert!(capped.len() <= 100);
}
