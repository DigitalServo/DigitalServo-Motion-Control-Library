//! Validation of identified models (`system_identification::validation`): information criteria,
//! residual tests (whiteness, cross-correlation, coherence, excited lines), comparison with the
//! nonparametric frequency response, one-step prediction errors, and `Validation::check`.

use dsmc::{DiscreteSystem, StateSpace, TransferFunction, TransferFunctionWithDelay, discretize::Zoh};

/// `Validation`: simulation of continuous / discrete models against the measured output, mean
/// squared residual and BIC.
#[test]
fn test_validation_bic() {
    use dsmc::system_identification::validation::{Validation, ValidationError};
    use dsmc::Discrete;
    use std::f64::consts::PI;

    // Uniform (0, 1) and standard normal pseudo-random numbers (xorshift64, Box-Muller)
    let mut state: u64 = 0x2545_f491_4f6c_dd1d;
    let mut uniform = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        ((state >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    };
    let mut gaussian = move || (-2.0 * uniform().ln()).sqrt() * (2.0 * PI * uniform()).cos();

    let ts: f64 = 1e-3;
    let n_samples = 10000;
    let delay = 3;
    let plant = TransferFunction::continuous(&[1000.0], &[1.0, 20.0, 1000.0]);
    let u: Vec<f64> = (0..n_samples).map(|_| gaussian()).collect();
    let mut system = DiscreteSystem::from(StateSpace::try_from(&plant).unwrap().discretize(Zoh, ts).unwrap());
    let y0: Vec<f64> = (0..n_samples)
        .map(|k| system.update(&[if k >= delay { u[k - delay] } else { 0.0 }]).unwrap()[0])
        .collect();

    // The true model reproduces the noise-free output
    let validation = Validation::continuous(TransferFunctionWithDelay::new(plant.clone(), delay as f64 * ts), ts, &u, &y0).unwrap();
    let max_residual = validation.residual.iter().fold(0.0, |acc: f64, v| acc.max(v.abs()));
    assert!(max_residual < 1e-10, "continuous: residual {max_residual:e}");

    // With white noise, V is the noise variance and BIC = N ln V + p ln N
    let sigma = 0.01;
    let y: Vec<f64> = y0.iter().map(|v| v + sigma * gaussian()).collect();
    let validation = Validation::continuous(TransferFunctionWithDelay::new(plant.clone(), delay as f64 * ts), ts, &u, &y).unwrap();
    let v = validation.mse();
    assert!((v / (sigma * sigma) - 1.0).abs() < 0.05, "mse {v:e}");
    let n = n_samples as f64;
    assert!((validation.bic(3) - (n * v.ln() + 3.0 * n.ln())).abs() < 1e-9);

    // A wrong delay or a missing pole is penalized
    let wrong_delay = Validation::continuous(TransferFunctionWithDelay::new(plant.clone(), (delay + 2) as f64 * ts), ts, &u, &y).unwrap();
    let first_order = TransferFunction::continuous(&[50.0], &[1.0, 50.0]);
    let wrong_order = Validation::continuous(TransferFunctionWithDelay::new(first_order.clone(), delay as f64 * ts), ts, &u, &y).unwrap();
    println!("BIC: true {:.1}, wrong delay {:.1}, first order {:.1}", validation.bic(3), wrong_delay.bic(3), wrong_order.bic(2));
    assert!(wrong_delay.bic(3) > validation.bic(3) + 10.0 * n.ln());
    assert!(wrong_order.bic(2) > validation.bic(3) + 10.0 * n.ln());

    // Evaluation on the second half only
    let second_half = validation.clone().evaluated_from(n_samples as f64 / 2.0 * ts).unwrap();
    assert_eq!(second_half.samples(), n_samples / 2);
    assert!((second_half.mse() / (sigma * sigma) - 1.0).abs() < 0.1);

    // Discrete-time model: y[k] = 1.5 y[k-1] - 0.7 y[k-2] + u[k-1] + 0.5 u[k-2]
    let g_z = TransferFunction::<f64, Discrete>::discrete(&[1.0, 0.5], &[1.0, -1.5, 0.7]);
    let mut y_arx = vec![0.0; n_samples];
    for k in 0..n_samples {
        let past = |x: &[f64], i: usize| if k >= i { x[k - i] } else { 0.0 };
        y_arx[k] = 1.5 * past(&y_arx, 1) - 0.7 * past(&y_arx, 2) + past(&u, 1) + 0.5 * past(&u, 2);
    }
    let validation = Validation::discrete(&g_z, ts, &u, &y_arx).unwrap();
    let max_residual = validation.residual.iter().fold(0.0, |acc: f64, v| acc.max(v.abs()));
    assert!(max_residual < 1e-10, "discrete: residual {max_residual:e}");

    // Errors
    let improper = TransferFunction::continuous(&[1.0, 0.0, 0.0], &[1.0, 1.0]);
    assert!(matches!(Validation::continuous(&improper, ts, &u, &y), Err(ValidationError::Simulation(dsmc::SimulationError::Improper { .. }))));
    assert!(matches!(Validation::continuous(&plant, ts, &u, &y[1..]), Err(ValidationError::LengthMismatch { .. })));
}

/// Cross-correlation test of the residual and the input: within the bound for the true model, far
/// outside for a wrong delay or a missing pole, with a white and with a colored (multisine) input.
#[test]
fn test_validation_cross_correlation() {
    use dsmc::system_identification::validation::Validation;
    use std::f64::consts::PI;

    let mut state: u64 = 0x9e37_79b9_7f4a_7c15;
    let mut uniform = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        ((state >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    };
    let mut gaussian = move || (-2.0 * uniform().ln()).sqrt() * (2.0 * PI * uniform()).cos();

    let ts: f64 = 1e-3;
    let n_samples = 20000;
    let delay = 3;
    let plant = TransferFunction::continuous(&[1000.0], &[1.0, 20.0, 1000.0]);
    let first_order = TransferFunction::continuous(&[50.0], &[1.0, 50.0]);
    let (max_lag, z) = (50, 2.58);

    let white: Vec<f64> = (0..n_samples).map(|_| gaussian()).collect();
    let phases: Vec<f64> = (0..40).map(|_| 2.0 * PI * uniform()).collect();
    let multisine: Vec<f64> = (0..n_samples)
        .map(|k| phases.iter().enumerate().map(|(i, p)| (2.0 * PI * (i + 1) as f64 * k as f64 * ts + p).sin()).sum::<f64>())
        .collect();

    for (name, u) in [("white", white), ("multisine", multisine)] {
        let mut system = DiscreteSystem::from(StateSpace::try_from(&plant).unwrap().discretize(Zoh, ts).unwrap());
        let y0: Vec<f64> = (0..n_samples)
            .map(|k| system.update(&[if k >= delay { u[k - delay] } else { 0.0 }]).unwrap()[0])
            .collect();
        let rms = (y0.iter().map(|v| v * v).sum::<f64>() / n_samples as f64).sqrt();
        let y: Vec<f64> = y0.iter().map(|v| v + 0.1 * rms * gaussian()).collect();

        let right = Validation::continuous(TransferFunctionWithDelay::new(plant.clone(), delay as f64 * ts), ts, &u, &y).unwrap().cross_correlation(max_lag, z);
        let wrong_delay = Validation::continuous(TransferFunctionWithDelay::new(plant.clone(), (delay + 1) as f64 * ts), ts, &u, &y).unwrap().cross_correlation(max_lag, z);
        let wrong_order = Validation::continuous(TransferFunctionWithDelay::new(first_order.clone(), delay as f64 * ts), ts, &u, &y).unwrap().cross_correlation(max_lag, z);
        for (model, c) in [("true", &right), ("wrong delay", &wrong_delay), ("first order", &wrong_order)] {
            println!("{name} input, {model}: bound {:.4}, outside {:.1} %, max ratio {:.2}", c.bound, 100.0 * c.fraction_outside(), c.max_ratio());
        }
        // 99 % per lag: a few percent outside at most for the true model
        assert!(right.fraction_outside() < 0.05 && right.max_ratio() < 1.5, "{name}: true model {:?}", right.outside());
        assert!(wrong_delay.max_ratio() > 2.0 && wrong_delay.fraction_outside() > 0.1, "{name}: wrong delay");
        assert!(wrong_order.max_ratio() > 2.0 && wrong_order.fraction_outside() > 0.1, "{name}: first order");
    }
}

/// Line test with a periodic multisine: the true model is within the bound with white and with
/// colored noise (the noise level is estimated per line), wrong models are far outside.
#[test]
fn test_validation_line_test() {
    use dsmc::system_identification::validation::{Validation, ValidationError};
    use std::f64::consts::PI;

    let mut state: u64 = 0x2545_f491_4f6c_dd1d;
    let mut uniform = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        ((state >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    };
    let mut gaussian = move || (-2.0 * uniform().ln()).sqrt() * (2.0 * PI * uniform()).cos();

    let ts: f64 = 1e-3;
    let period = 1000; // 1 s: lines at 1 Hz spacing
    let period_s = period as f64 * ts;
    let f0 = 1.0 / period_s;
    let lines: Vec<usize> = (1..=40).collect();
    let n_samples = 6 * period; // the first period (transient) is not evaluated: P = 5
    let delay = 3;
    let plant = TransferFunction::continuous(&[1000.0], &[1.0, 20.0, 1000.0]);
    let first_order = TransferFunction::continuous(&[50.0], &[1.0, 50.0]);

    let phases: Vec<f64> = lines.iter().map(|_| 2.0 * PI * uniform()).collect();
    let u: Vec<f64> = (0..n_samples)
        .map(|k| lines.iter().zip(&phases).map(|(&l, p)| (2.0 * PI * l as f64 * k as f64 / period as f64 + p).sin()).sum::<f64>())
        .collect();
    let mut system = DiscreteSystem::from(StateSpace::try_from(&plant).unwrap().discretize(Zoh, ts).unwrap());
    let y0: Vec<f64> = (0..n_samples)
        .map(|k| system.update(&[if k >= delay { u[k - delay] } else { 0.0 }]).unwrap()[0])
        .collect();
    let rms = (y0.iter().map(|v| v * v).sum::<f64>() / n_samples as f64).sqrt();

    // White noise, and noise of the same power low-passed at ~5 Hz (concentrated at the low lines)
    let white: Vec<f64> = (0..n_samples).map(|_| 0.1 * rms * gaussian()).collect();
    let mut colored = vec![0.0; n_samples];
    for k in 1..n_samples {
        colored[k] = 0.97 * colored[k - 1] + (1.0 - 0.97f64 * 0.97).sqrt() * 0.1 * rms * gaussian();
    }

    for (name, noise) in [("white", &white), ("colored", &colored)] {
        let y: Vec<f64> = y0.iter().zip(noise.iter()).map(|(a, b)| a + b).collect();
        let test = |model: &TransferFunction<f64>, nk: usize| {
            Validation::continuous(TransferFunctionWithDelay::new(model.clone(), nk as f64 * ts), ts, &u, &y).unwrap().evaluated_from(period_s).unwrap().line_test(f0, &lines, 0.99).unwrap()
        };
        let (right, wrong_delay, wrong_order) = (test(&plant, delay), test(&plant, delay + 1), test(&first_order, delay));
        for (model, t) in [("true", &right), ("wrong delay", &wrong_delay), ("first order", &wrong_order)] {
            println!(
                "{name} noise, {model}: P = {}, bound {:.2}, outside {:.1} %, mean F {:.2} (expected {:.2}), relative error {:.4}",
                t.periods, t.bound, 100.0 * t.fraction_outside(), t.mean_statistic(), t.expected_statistic(), t.rms_relative_error()
            );
        }
        assert_eq!(right.periods, 5);
        assert!(right.fraction_outside() <= 0.05 && right.mean_statistic() < 2.0 * right.expected_statistic(), "{name}: true model");
        // One sample of delay matters at 5 .. 20 Hz, where the colored noise is ~3 times the white
        // one: there the shift is within the noise, and the test rightly does not reject it.
        if name == "white" {
            assert!(wrong_delay.fraction_outside() > 0.15 && wrong_delay.mean_statistic() > 3.0 * right.expected_statistic(), "{name}: wrong delay");
        }
        assert!(wrong_order.fraction_outside() > 0.5 && wrong_order.mean_statistic() > 10.0 * right.expected_statistic(), "{name}: first order");
    }

    // Errors
    let validation = Validation::continuous(TransferFunctionWithDelay::new(plant.clone(), delay as f64 * ts), ts, &u, &y0).unwrap();
    assert!(matches!(validation.clone().evaluated_from(5.0 * period_s).unwrap().line_test(f0, &lines, 0.99), Err(ValidationError::TooFewPeriods { periods: 1 })));
    assert!(matches!(validation.line_test(f0, &[600], 0.99), Err(ValidationError::InvalidLine { line: 600, .. })));
}

/// Coherence test of the residual and the input with non-periodic inputs (chirp, low-passed random
/// signal) and colored noise: the true model is within the bound at about the nominal rate, wrong
/// models are far outside.
#[test]
fn test_validation_coherence_test() {
    use dsmc::system_identification::validation::{Validation, ValidationError};
    use std::f64::consts::PI;

    let mut state: u64 = 0x2545_f491_4f6c_dd1d;
    let mut uniform = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        ((state >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    };
    let mut gaussian = move || (-2.0 * uniform().ln()).sqrt() * (2.0 * PI * uniform()).cos();

    let ts: f64 = 1e-3;
    let n_samples = 40000;
    let delay = 3;
    let plant = TransferFunction::continuous(&[1000.0], &[1.0, 20.0, 1000.0]);
    let first_order = TransferFunction::continuous(&[50.0], &[1.0, 50.0]);
    let segment_s = 1.0; // 1 Hz resolution

    // Linear chirp 0.5 .. 50 Hz over the record, and random noise low-passed at ~30 Hz
    let duration = n_samples as f64 * ts;
    let chirp: Vec<f64> = (0..n_samples)
        .map(|k| {
            let t = k as f64 * ts;
            (2.0 * PI * (0.5 * t + (50.0 - 0.5) * t * t / (2.0 * duration))).sin()
        })
        .collect();
    let mut random = vec![0.0; n_samples];
    for k in 1..n_samples {
        random[k] = 0.83 * random[k - 1] + gaussian();
    }

    for (name, u) in [("chirp", chirp), ("random", random)] {
        let mut system = DiscreteSystem::from(StateSpace::try_from(&plant).unwrap().discretize(Zoh, ts).unwrap());
        let y0: Vec<f64> = (0..n_samples)
            .map(|k| system.update(&[if k >= delay { u[k - delay] } else { 0.0 }]).unwrap()[0])
            .collect();
        let rms = (y0.iter().map(|v| v * v).sum::<f64>() / n_samples as f64).sqrt();
        // Colored noise (low-passed white noise), 10 % of the output RMS
        let mut noise = vec![0.0; n_samples];
        for k in 1..n_samples {
            noise[k] = 0.9 * noise[k - 1] + (1.0 - 0.81f64).sqrt() * 0.1 * rms * gaussian();
        }
        let y: Vec<f64> = y0.iter().zip(&noise).map(|(a, b)| a + b).collect();

        let test = |model: &TransferFunction<f64>, nk: usize, confidence: f64| {
            Validation::continuous(TransferFunctionWithDelay::new(model.clone(), nk as f64 * ts), ts, &u, &y).unwrap().coherence_test(segment_s, confidence).unwrap().excited(1e-2)
        };
        let (right, wrong_delay, wrong_order) = (test(&plant, delay, 0.99), test(&plant, delay + 1, 0.99), test(&first_order, delay, 0.99));
        for (model, t) in [("true", &right), ("wrong delay", &wrong_delay), ("first order", &wrong_order)] {
            println!(
                "{name}, {model}: K = {}, L = {:.1}, {} bins, bound {:.4}, outside {:.1} %, mean γ² {:.4} (expected {:.4})",
                t.segments, t.effective_segments, t.bins.len(), t.bound, 100.0 * t.fraction_outside(), t.mean_coherence(), t.expected_coherence()
            );
        }
        println!("{name}, wrong delay: outside at {:?} Hz", wrong_delay.outside());
        assert!(right.fraction_outside() < 0.05 && right.mean_coherence() < 1.5 * right.expected_coherence(), "{name}: true model");
        // One sample of delay shows where the plant has gain (the resonance at 5 Hz and around);
        // the random input also excites up to ~300 Hz, where it is buried in the noise, so the
        // fraction is taken up to 50 Hz
        let low = wrong_delay.outside().iter().filter(|&&f| f <= 50).count() as f64
            / wrong_delay.bins.iter().filter(|&&f| f <= 50).count() as f64;
        assert!(low > 0.15 && wrong_delay.outside().iter().filter(|&&f| (3..=11).contains(&f)).count() >= 8, "{name}: wrong delay");
        assert!(wrong_order.fraction_outside() > 0.3 && wrong_order.mean_coherence() > 5.0 * right.expected_coherence(), "{name}: first order");

        // Calibration of the bound for the true model: about 5 % of the bins outside at 95 %
        let at_95 = test(&plant, delay, 0.95);
        println!("{name}, true at 95 %: outside {:.1} %", 100.0 * at_95.fraction_outside());
        assert!(at_95.fraction_outside() < 0.12, "{name}: {:.3} outside at 95 %", at_95.fraction_outside());
    }

    // Errors
    let u = vec![0.0; 1400]; // one segment of 1000 (the next would start at 500)
    let validation = Validation::continuous(&plant, ts, &u, &u).unwrap();
    assert!(matches!(validation.coherence_test(1.0, 0.99), Err(ValidationError::TooFewSegments { segments: 1, .. })));
}

/// Frequency response of the model against the nonparametric estimate (random input, colored
/// noise): the error is at the noise level for the true model, the phase error of a wrong delay is
/// `-ω ts`, and a missing pole is far beyond the noise.
#[test]
fn test_validation_frequency_response() {
    use dsmc::system_identification::validation::Validation;
    use nalgebra::Complex;
    use std::f64::consts::PI;

    let mut state: u64 = 0x9e37_79b9_7f4a_7c15;
    let mut uniform = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        ((state >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    };
    let mut gaussian = move || (-2.0 * uniform().ln()).sqrt() * (2.0 * PI * uniform()).cos();

    let ts: f64 = 1e-3;
    let n_samples = 40000;
    let delay = 3;
    let plant = TransferFunction::continuous(&[1000.0], &[1.0, 20.0, 1000.0]);
    let first_order = TransferFunction::continuous(&[50.0], &[1.0, 50.0]);
    let segment_s = 1.0;

    let mut u = vec![0.0; n_samples];
    for k in 1..n_samples {
        u[k] = 0.83 * u[k - 1] + gaussian();
    }
    let mut system = DiscreteSystem::from(StateSpace::try_from(&plant).unwrap().discretize(Zoh, ts).unwrap());
    let y0: Vec<f64> = (0..n_samples)
        .map(|k| system.update(&[if k >= delay { u[k - delay] } else { 0.0 }]).unwrap()[0])
        .collect();
    let rms = (y0.iter().map(|v| v * v).sum::<f64>() / n_samples as f64).sqrt();
    let mut noise = vec![0.0; n_samples];
    for k in 1..n_samples {
        noise[k] = 0.9 * noise[k - 1] + (1.0 - 0.81f64).sqrt() * 0.1 * rms * gaussian();
    }
    let y: Vec<f64> = y0.iter().zip(&noise).map(|(a, b)| a + b).collect();

    let compare = |model: &TransferFunction<f64>, nk: usize| {
        Validation::continuous(TransferFunctionWithDelay::new(model.clone(), nk as f64 * ts), ts, &u, &y).unwrap().frequency_response(segment_s).unwrap().excited(1e-2)
    };
    let (right, wrong_delay, wrong_order) = (compare(&plant, delay), compare(&plant, delay + 1), compare(&first_order, delay));
    for (name, c) in [("true", &right), ("wrong delay", &wrong_delay), ("first order", &wrong_order)] {
        println!(
            "{name}: {} bins, L = {:.1}, rms relative error {:.4}, mean normalized error {:.2}",
            c.bins.len(), c.effective_segments, c.rms_relative_error(), c.mean_normalized_error()
        );
    }

    // True model: the error is at the noise level, and small where the data are accurate
    // (output-input coherence above 0.9)
    let mean = right.mean_normalized_error();
    assert!(mean > 0.5 && mean < 2.0, "true model: mean normalized error {mean}");
    let accurate: Vec<usize> = (0..right.bins.len()).filter(|&i| right.coherence[i] > 0.9).collect();
    println!("accurate data at {:?} Hz", accurate.iter().map(|&i| right.bins[i]).collect::<Vec<_>>());
    assert!(accurate.len() >= 10);
    for &i in &accurate {
        assert!(right.gain_error_db()[i].abs() < 0.5 && right.phase_error()[i].abs() < 0.05, "{} Hz: true model", right.bins[i]);
    }

    // The model response is a windowed estimate: compared with the exact response of the sampled
    // system (the ZOH-discretized G(z) at z = e^(jω ts), with the delay; not G(jω), which lacks the
    // half-sample delay of the hold), its bias falls as the segments grow against the impulse
    // response of the plant (~0.1 s here)
    let g_z = plant.discretize(Zoh, ts).unwrap();
    let bias = |segment_len: usize| {
        let c = Validation::continuous(TransferFunctionWithDelay::new(plant.clone(), delay as f64 * ts), ts, &u, &y).unwrap().frequency_response(segment_len as f64 * ts).unwrap();
        c.frequencies()
            .iter()
            .enumerate()
            .filter(|&(_, &f)| (2.0..=20.0).contains(&f))
            .map(|(i, &f)| {
                let z = Complex::new(0.0, 2.0 * PI * f * ts).exp();
                let eval = |c: &[f64]| c.iter().fold(Complex::new(0.0, 0.0), |acc, &x| acc * z + x);
                let exact = eval(&g_z.numerator) / eval(&g_z.denominator) * z.powi(-(delay as i32));
                (c.model[i] - exact).norm() / exact.norm()
            })
            .fold(0.0, f64::max)
    };
    let (coarse, fine) = (bias(1000), bias(4000));
    println!("bias of the model response over 2 .. 20 Hz: {coarse:.4} (1 s segments), {fine:.4} (4 s)");
    assert!(fine < 0.05 && fine < 0.5 * coarse, "bias {coarse} -> {fine}");

    // One sample of delay: phase error -ω ts, gain unchanged, where the data are accurate
    let frequencies = wrong_delay.frequencies();
    for &i in &accurate {
        let f = frequencies[i];
        let expected = -2.0 * PI * f * ts;
        assert!((wrong_delay.phase_error()[i] - expected).abs() < 0.2 * expected.abs() + 0.01, "{f} Hz: phase error {}", wrong_delay.phase_error()[i]);
        assert!(wrong_delay.gain_error_db()[i].abs() < 0.5, "{f} Hz: gain error {}", wrong_delay.gain_error_db()[i]);
    }
    // ... where it is beyond the noise (over all bins it is diluted by the high frequencies, where
    // the plant has no gain). Around the resonance the leakage of the 1 s segments lowers the
    // coherence and so inflates σ: there the normalized error of the true model is below its
    // nominal mean 1 (conservative)
    let at_accurate = |c: &dsmc::system_identification::validation::FrequencyResponseComparison<f64>| {
        let e = c.normalized_error();
        accurate.iter().map(|&i| e[i]).sum::<f64>() / accurate.len() as f64
    };
    println!("mean normalized error where accurate: true {:.2}, wrong delay {:.2}", at_accurate(&right), at_accurate(&wrong_delay));
    assert!(at_accurate(&wrong_delay) > 1.0 && at_accurate(&wrong_delay) > 5.0 * at_accurate(&right));

    // A missing pole is far beyond the uncertainty of the data
    assert!(wrong_order.mean_normalized_error() > 100.0 && wrong_order.rms_relative_error() > 0.5);
}

/// AIC, AICc and BIC: definitions, and their penalties against each other.
#[test]
fn test_validation_aic() {
    use dsmc::system_identification::validation::Validation;

    let plant = TransferFunction::continuous(&[1000.0], &[1.0, 20.0, 1000.0]);
    let u: Vec<f64> = (0..1000).map(|k| ((k * 7919) % 1000) as f64 / 500.0 - 1.0).collect();
    let y: Vec<f64> = (0..1000).map(|k| 0.01 * (((k * 104729) % 997) as f64 / 498.5 - 1.0)).collect();
    let validation = Validation::continuous(&plant, 1e-3, &u, &y).unwrap();
    let (n, v) = (1000.0, validation.mse());

    assert!((validation.aic(3) - (n * v.ln() + 6.0)).abs() < 1e-9);
    assert!((validation.aicc(3) - (validation.aic(3) + 24.0 / 996.0)).abs() < 1e-9);
    assert!((validation.bic(3) - (n * v.ln() + 3.0 * n.ln())).abs() < 1e-9);
    // One more parameter costs 2 in AIC and ln N ≈ 6.9 in BIC
    assert!((validation.aic(4) - validation.aic(3) - 2.0).abs() < 1e-9);
    assert!((validation.bic(4) - validation.bic(3) - n.ln()).abs() < 1e-9);
    // AICc diverges when the parameters approach the samples
    let short = validation.clone().evaluated_from(0.995).unwrap();
    assert!(short.aicc(3).is_finite() && short.aicc(4).is_infinite());
}

/// Whiteness test of the residual: a white residual passes, a colored one fails. For an output
/// error model with colored noise, the true model fails the whiteness test (the noise is colored)
/// but passes the cross-correlation test (`G` is right).
#[test]
fn test_validation_whiteness() {
    use dsmc::system_identification::validation::Validation;
    use std::f64::consts::PI;

    let mut state: u64 = 0x2545_f491_4f6c_dd1d;
    let mut uniform = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        ((state >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    };
    let mut gaussian = move || (-2.0 * uniform().ln()).sqrt() * (2.0 * PI * uniform()).cos();

    let ts: f64 = 1e-3;
    let n_samples = 20000;
    let delay = 3;
    let plant = TransferFunction::continuous(&[1000.0], &[1.0, 20.0, 1000.0]);
    let first_order = TransferFunction::continuous(&[50.0], &[1.0, 50.0]);
    let (max_lag, z) = (20, 2.58);
    let z_one_sided = 2.326; // 99 %

    let u: Vec<f64> = (0..n_samples).map(|_| gaussian()).collect();
    let mut system = DiscreteSystem::from(StateSpace::try_from(&plant).unwrap().discretize(Zoh, ts).unwrap());
    let y0: Vec<f64> = (0..n_samples)
        .map(|k| system.update(&[if k >= delay { u[k - delay] } else { 0.0 }]).unwrap()[0])
        .collect();
    let rms = (y0.iter().map(|v| v * v).sum::<f64>() / n_samples as f64).sqrt();
    let white: Vec<f64> = (0..n_samples).map(|_| 0.1 * rms * gaussian()).collect();
    let mut colored = vec![0.0; n_samples];
    for k in 1..n_samples {
        colored[k] = 0.9 * colored[k - 1] + (1.0 - 0.81f64).sqrt() * 0.1 * rms * gaussian();
    }

    let check = |model: &TransferFunction<f64>, noise: &[f64]| {
        let y: Vec<f64> = y0.iter().zip(noise).map(|(a, b)| a + b).collect();
        let validation = Validation::continuous(TransferFunctionWithDelay::new(model.clone(), delay as f64 * ts), ts, &u, &y).unwrap();
        (validation.autocorrelation(max_lag, z), validation.cross_correlation(50, z))
    };

    // Wilson-Hilferty against the exact 99 % quantile of χ²(20), 37.566
    let (white_true, white_cross) = check(&plant, &white);
    assert!((white_true.ljung_box_bound(z_one_sided) - 37.566).abs() < 0.2, "{}", white_true.ljung_box_bound(z_one_sided));

    let (colored_true, colored_cross) = check(&plant, &colored);
    let (white_wrong, _) = check(&first_order, &white);
    for (name, w) in [("true, white noise", &white_true), ("true, colored noise", &colored_true), ("first order, white noise", &white_wrong)] {
        println!(
            "{name}: outside {:.0} %, max ratio {:.2}, Ljung-Box {:.1} (bound {:.1})",
            100.0 * w.fraction_outside(), w.max_ratio(), w.ljung_box, w.ljung_box_bound(z_one_sided)
        );
    }

    // White noise, true model: white
    assert!(white_true.fraction_outside() <= 0.1 && white_true.ljung_box <= white_true.ljung_box_bound(z_one_sided));
    assert!(white_cross.fraction_outside() < 0.05);
    // Colored noise, true model: not white, but independent of the input
    assert!(colored_true.max_ratio() > 10.0 && colored_true.ljung_box > 10.0 * colored_true.ljung_box_bound(z_one_sided));
    assert!(colored_cross.fraction_outside() < 0.05, "colored noise: cross-correlation {:?}", colored_cross.outside());
    // Wrong model: the unmodeled response (G - Ĝ) u makes the residual colored
    assert!(white_wrong.ljung_box > 10.0 * white_wrong.ljung_box_bound(z_one_sided));
}

/// One-step prediction error of ARX models: white for data with a white equation error (the ARX
/// noise model holds), colored for white output noise even with the true parameters, where the
/// IV estimate validated by its output error passes instead.
#[test]
fn test_validation_one_step_prediction() {
    use dsmc::system_identification::{arx::Arx, iv, lsm, validation::Validation};
    use nalgebra::DVector;
    use std::f64::consts::PI;

    let mut state: u64 = 0x9e37_79b9_7f4a_7c15;
    let mut uniform = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        ((state >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    };
    let mut gaussian = move || (-2.0 * uniform().ln()).sqrt() * (2.0 * PI * uniform()).cos();

    // y[k] = 1.5 y[k-1] - 0.7 y[k-2] + u[k-1] + 0.5 u[k-2] (+ noise): na = 2, nb = 1, nk = 1
    let n_samples = 20000;
    let structure = Arx::<f64>::new(2, 1).with_input_delay(1);
    let mut truth = structure.clone();
    truth.parameter = DVector::from_vec(vec![1.5, -0.7, 1.0, 0.5]);
    let simulate = |u: &[f64], e: &[f64]| {
        let mut y = vec![0.0; u.len()];
        for k in 0..u.len() {
            let past = |x: &[f64], i: usize| if k >= i { x[k - i] } else { 0.0 };
            y[k] = 1.5 * past(&y, 1) - 0.7 * past(&y, 2) + past(u, 1) + 0.5 * past(u, 2) + e[k];
        }
        y
    };
    let least_squares = |structure: &Arx<f64>, u: &[f64], y: &[f64]| {
        let mut buffer = lsm::arx::DataBuffer::from_arx(structure.clone());
        for k in 0..u.len() {
            buffer.add(u[k], if k > 0 { y[k - 1] } else { 0.0 }, y[k]);
        }
        buffer.identify().unwrap();
        buffer.arx
    };
    let (max_lag, z, z_one_sided) = (20, 2.58, 2.326);
    let white = |v: &Validation<f64>| {
        let w = v.autocorrelation(max_lag, z);
        (w.ljung_box, w.ljung_box_bound(z_one_sided))
    };

    let u: Vec<f64> = (0..n_samples).map(|_| gaussian()).collect();
    let e: Vec<f64> = (0..n_samples).map(|_| 0.1 * gaussian()).collect();

    // Equation error: the prediction error of the true model is e itself
    let y = simulate(&u, &e);
    let validation = Validation::one_step_prediction(&truth, 1e-3, &u, &y).unwrap();
    let max_difference = validation.residual.iter().zip(&e).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
    assert!(max_difference < 1e-12, "prediction error of the true model differs from e by {max_difference:e}");

    // ... and the least-squares estimate is white and independent of the input
    let estimate = Validation::one_step_prediction(&least_squares(&structure, &u, &y), 1e-3, &u, &y).unwrap();
    let (q, bound) = white(&estimate);
    println!("equation error, LS: Ljung-Box {q:.1} (bound {bound:.1}), mse {:.4e}", estimate.mse());
    assert!(q <= bound && estimate.cross_correlation(50, z).fraction_outside() < 0.05);
    assert!((estimate.mse() / 0.01 - 1.0).abs() < 0.05);

    // ... while a missing pole leaves a colored prediction error
    let low = Validation::one_step_prediction(&least_squares(&Arx::new(1, 1).with_input_delay(1), &u, &y), 1e-3, &u, &y).unwrap();
    let (q_low, _) = white(&low);
    println!("equation error, na = 1: Ljung-Box {q_low:.1}");
    assert!(q_low > 10.0 * bound);

    // Output error y = G u + v: even the true A, B leave the colored error A(z) v
    let y0 = simulate(&u, &vec![0.0; n_samples]);
    let y: Vec<f64> = y0.iter().map(|v| v + 0.3 * gaussian()).collect();
    let (q_true, _) = white(&Validation::one_step_prediction(&truth, 1e-3, &u, &y).unwrap());
    let (q_ls, _) = white(&Validation::one_step_prediction(&least_squares(&structure, &u, &y), 1e-3, &u, &y).unwrap());
    // ... the IV estimate is validated by its output error instead: white, as v
    let iv_model = iv::arx::identify(&u, &y, &structure, 3).unwrap();
    let output_error = Validation::discrete(&iv_model.transfer_function(), 1e-3, &u, &y).unwrap();
    let (q_iv, _) = white(&output_error);
    println!("output error: prediction error of the true model {q_true:.1}, of LS {q_ls:.1}; output error of IV {q_iv:.1}");
    assert!(q_true > 10.0 * bound && q_ls > 10.0 * bound);
    assert!(q_iv <= bound && output_error.cross_correlation(50, z).fraction_outside() < 0.05);
}

/// `Validation::check`: several tests at one confidence level with pass / fail rules. The true
/// model passes, a wrong delay fails the input tests, colored noise fails the whiteness test only,
/// and the true model is rejected by chance at about the nominal rate.
#[test]
fn test_validation_check() {
    use dsmc::system_identification::validation::{Check, CheckResult, CoherenceCheck, Validation};
    use std::f64::consts::PI;

    let mut state: u64 = 0x2545_f491_4f6c_dd1d;
    let mut uniform = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        ((state >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    };
    let mut gaussian = move || (-2.0 * uniform().ln()).sqrt() * (2.0 * PI * uniform()).cos();

    let ts: f64 = 1e-3;
    let period = 1000;
    let n_samples = 6 * period;
    let delay = 3;
    let plant = TransferFunction::continuous(&[1000.0], &[1.0, 20.0, 1000.0]);
    let lines: Vec<usize> = (1..=40).collect();
    // Periodic multisine (also used by the non-periodic tests)
    let phases: Vec<f64> = lines.iter().map(|_| 2.0 * PI * uniform()).collect();
    let u: Vec<f64> = (0..n_samples)
        .map(|k| lines.iter().zip(&phases).map(|(&l, p)| (2.0 * PI * l as f64 * k as f64 / period as f64 + p).sin()).sum::<f64>())
        .collect();
    let mut system = DiscreteSystem::from(StateSpace::try_from(&plant).unwrap().discretize(Zoh, ts).unwrap());
    let y0: Vec<f64> = (0..n_samples)
        .map(|k| system.update(&[if k >= delay { u[k - delay] } else { 0.0 }]).unwrap()[0])
        .collect();
    let rms = (y0.iter().map(|v| v * v).sum::<f64>() / n_samples as f64).sqrt();

    let checks = [
        Check::InformationCriteria { parameters: 3 },
        Check::Whiteness { max_lag: 20 },
        Check::CrossCorrelation { max_lag: 50 },
        Check::Coherence(CoherenceCheck::new(0.5, 1e-2)),
        Check::Lines { fundamental_frequency: 1.0 / (period as f64 * ts), lines: lines.clone() },
    ];
    let mut noisy = |colored: bool| {
        let mut v = 0.0;
        y0.iter()
            .map(|y| {
                v = if colored { 0.9 * v + (1.0 - 0.81f64).sqrt() * 0.1 * rms * gaussian() } else { 0.1 * rms * gaussian() };
                y + v
            })
            .collect::<Vec<f64>>()
    };
    let run = |nk: usize, y: &[f64]| {
        Validation::continuous(TransferFunctionWithDelay::new(plant.clone(), nk as f64 * ts), ts, &u, y).unwrap().evaluated_from(period as f64 * ts).unwrap().check(&checks, 0.99).unwrap()
    };
    let passed = |report: &dsmc::system_identification::validation::Report<f64>| -> Vec<Option<bool>> {
        report.results.iter().map(CheckResult::passed).collect()
    };

    let white = noisy(false);
    let right = run(delay, &white);
    println!("true model, white noise:\n{right}");
    assert!(right.passed() && passed(&right) == [None, Some(true), Some(true), Some(true), Some(true)]);

    let wrong = run(delay + 1, &white);
    println!("wrong delay:\n{wrong}");
    assert!(!wrong.passed());
    assert_eq!(passed(&wrong)[2..4], [Some(false), Some(false)], "cross-correlation and coherence");

    let colored = run(delay, &noisy(true));
    println!("true model, colored noise:\n{colored}");
    assert_eq!(passed(&colored), [None, Some(false), Some(true), Some(true), Some(true)]);

    // False rejections of the true model over independent noise records. Measured over 400
    // records (release build): whiteness 3, cross-correlation 0 (Bonferroni, conservative with
    // correlated lags; counting the lags outside against a binomial quantile instead gave 48),
    // coherence 1, lines 5, i.e. about the nominal 1 % per test.
    let trials = 40;
    let rejected = (0..trials).filter(|_| !run(delay, &noisy(false)).passed()).count();
    println!("true model rejected in {rejected} of {trials} trials (4 tests at 99 %)");
    assert!(rejected <= 6, "{rejected} of {trials}");
}

/// Rigid-body model of a plant with a resonance above the control bandwidth: the coherence test
/// fails on the whole band (the residual keeps the error of the resonance), and passes on the
/// band `set_band` below it, where the model error is at the noise level.
#[test]
fn test_validation_coherence_band() {
    use dsmc::system_identification::validation::{Check, CoherenceCheck, Validation, ValidationError};
    use std::f64::consts::PI;

    let mut state: u64 = 0x2545_f491_4f6c_dd1d;
    let mut uniform = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        ((state >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    };
    let mut gaussian = move || (-2.0 * uniform().ln()).sqrt() * (2.0 * PI * uniform()).cos();

    let ts: f64 = 1e-3;
    let n_samples = 40000;
    // Rigid body 10 / (s + 10) with an antiresonance at 150 Hz and a resonance at 200 Hz (unit DC
    // gain). Their quasi-static tail is a gain error `(f / 150)^2 - (f / 200)^2` below them (0.2 %
    // at 10 Hz), within the noise of the data in the band; with the modes at 60 / 90 Hz (1.6 %)
    // the band test rightly fails.
    let rigid = TransferFunction::continuous(&[10.0], &[1.0, 10.0]);
    let (wa, wr, zeta) = (2.0 * PI * 150.0, 2.0 * PI * 200.0, 0.02);
    let mode = TransferFunction::continuous(&[wr * wr / (wa * wa), 2.0 * zeta * wr * wr / wa, wr * wr], &[1.0, 2.0 * zeta * wr, wr * wr]);
    let plant = &rigid * &mode;

    let u: Vec<f64> = (0..n_samples).map(|_| gaussian()).collect();
    let y0 = Validation::continuous(&plant, ts, &u, &u).unwrap().simulated;
    let rms = (y0.iter().map(|v| v * v).sum::<f64>() / n_samples as f64).sqrt();
    let y: Vec<f64> = y0.iter().map(|v| v + 0.05 * rms * gaussian()).collect();
    let validation = Validation::continuous(&rigid, ts, &u, &y).unwrap();

    let band = (0.0, 10.0);
    let whole = Check::Coherence(CoherenceCheck::new(1.0, 1e-2));
    let low = Check::Coherence(CoherenceCheck::new(1.0, 1e-2).set_band(band));
    let report = validation.check(&[whole, low], 0.99).unwrap();
    println!("rigid-body model:\n{report}");
    assert_eq!(report.results.iter().map(|r| r.passed()).collect::<Vec<_>>(), [Some(false), Some(true)]);

    // The bins of the band, both edges included (1 Hz bins)
    let coherence = validation.coherence_test(1.0, 0.99).unwrap().band(band);
    assert_eq!(coherence.bins, (1..=10).collect::<Vec<_>>());
    let response = validation.frequency_response(1.0).unwrap();
    let in_band = response.band(band);
    assert_eq!(in_band.bins, coherence.bins);
    println!("relative error: {:.4} in the band, {:.4} on the whole band", in_band.rms_relative_error(), response.rms_relative_error());
    assert!(in_band.rms_relative_error() < 0.02 && response.rms_relative_error() > 0.5);

    let invalid = Check::Coherence(CoherenceCheck::new(1.0, 1e-2).set_band((10.0, 5.0)));
    assert_eq!(validation.check(&[invalid], 0.99).unwrap_err(), ValidationError::InvalidBand { low: 10.0, high: 5.0 });
}

/// The tests do not depend on the units of the signals (input and output scaled by 1e-4 and
/// 1e-5, the model by their ratio), and a residual or an input that is identically zero gives zero
/// statistics rather than NaN.
#[test]
fn test_validation_scale_and_zero_signals() {
    use dsmc::system_identification::validation::Validation;
    use std::f64::consts::PI;

    let mut state: u64 = 0x2545_f491_4f6c_dd1d;
    let mut uniform = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        ((state >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    };
    let mut gaussian = move || (-2.0 * uniform().ln()).sqrt() * (2.0 * PI * uniform()).cos();

    let ts: f64 = 1e-3;
    let n_samples = 20000;
    let plant = TransferFunction::continuous(&[1000.0], &[1.0, 20.0, 1000.0]);
    let mut u = vec![0.0; n_samples];
    for k in 1..n_samples {
        u[k] = 0.83 * u[k - 1] + gaussian();
    }
    let y0 = Validation::continuous(&plant, ts, &u, &vec![0.0; n_samples]).unwrap().simulated;
    let y: Vec<f64> = y0.iter().map(|v| v + 0.05 * gaussian()).collect();
    // Wrong order, so that the residual depends on the input
    let model = TransferFunction::continuous(&[50.0], &[1.0, 50.0]);

    let (cu, cy) = (1e-4, 1e-5);
    let (u_small, y_small): (Vec<f64>, Vec<f64>) = (u.iter().map(|v| v * cu).collect(), y.iter().map(|v| v * cy).collect());
    let model_small = TransferFunction::continuous(&[50.0 * cy / cu], &[1.0, 50.0]);
    let unit = Validation::continuous(&model, ts, &u, &y).unwrap();
    let small = Validation::continuous(&model_small, ts, &u_small, &y_small).unwrap();

    let close = |a: &[f64], b: &[f64], what: &str| {
        assert_eq!(a.len(), b.len());
        for (i, (x, z)) in a.iter().zip(b).enumerate() {
            assert!((x - z).abs() <= 1e-9 * x.abs().max(1e-12), "{what}[{i}]: {x} vs {z}");
        }
    };
    let (c1, c2) = (unit.coherence_test(0.5, 0.99).unwrap(), small.coherence_test(0.5, 0.99).unwrap());
    close(&c1.coherence, &c2.coherence, "coherence");
    assert!(c1.mean_coherence() > 5.0 * c1.expected_coherence());
    let (f1, f2) = (unit.frequency_response(0.5).unwrap(), small.frequency_response(0.5).unwrap());
    close(&f1.coherence, &f2.coherence, "frequency response coherence");
    close(&f1.relative_error(), &f2.relative_error(), "relative error");
    let (x1, x2) = (unit.cross_correlation(20, 3.0), small.cross_correlation(20, 3.0));
    close(&x1.correlation, &x2.correlation, "cross-correlation");
    close(&[x1.bound], &[x2.bound], "cross-correlation bound");
    assert!(!x1.outside().is_empty());
    let (w1, w2) = (unit.autocorrelation(20, 2.58), small.autocorrelation(20, 2.58));
    close(&w1.correlation, &w2.correlation, "autocorrelation");

    // Residual identically zero: the output is the simulated output itself
    let exact = Validation::continuous(&plant, ts, &u, &y0).unwrap();
    assert!(exact.residual.iter().all(|&e| e == 0.0));
    let w = exact.autocorrelation(20, 2.58);
    assert!(w.correlation.iter().all(|&r| r == 0.0) && w.ljung_box == 0.0, "{w:?}");
    let x = exact.cross_correlation(20, 3.0);
    assert!(x.correlation.iter().all(|&r| r == 0.0) && x.bound.is_finite() && x.outside().is_empty(), "{x:?}");
    let c = exact.coherence_test(0.5, 0.99).unwrap();
    assert!(c.coherence.iter().all(|&g| g == 0.0) && c.outside().is_empty());
    let l = exact.line_test(2.0, &[1, 2, 3], 0.99).unwrap();
    assert!(l.statistic.iter().all(|&f| f == 0.0), "{l:?}");

    // Input identically zero: nothing to correlate, no NaN
    let silent = Validation::continuous(&plant, ts, &vec![0.0; n_samples], &y).unwrap();
    let x = silent.cross_correlation(20, 3.0);
    assert!(x.correlation.iter().all(|&r| r == 0.0) && x.bound.is_finite(), "{x:?}");
    assert!(silent.frequency_response(0.5).unwrap().bins.is_empty());
    assert!(silent.coherence_test(0.5, 0.99).unwrap().bins.is_empty());

    // Band-limited input (lines 1 ..= 50 of the segment): the bins without input power are left
    // out, so the errors are finite at every bin returned
    let band: Vec<f64> = (0..n_samples)
        .map(|k| (1..=50).map(|l| (2.0 * PI * (l * k) as f64 / 500.0 + 0.3 * (l * l) as f64).cos()).sum())
        .collect();
    let y_band: Vec<f64> = Validation::continuous(&plant, ts, &band, &vec![0.0; n_samples]).unwrap().simulated.iter().map(|v| v + 0.01 * gaussian()).collect();
    let limited = Validation::continuous(&model, ts, &band, &y_band).unwrap();
    let f = limited.frequency_response(0.5).unwrap();
    let c = limited.coherence_test(0.5, 0.99).unwrap();
    for bins in [&f.bins, &c.bins] {
        assert!((1..=50).all(|l| bins.contains(&l)), "{bins:?}");
        assert!(bins.iter().all(|&b| b < 200), "{bins:?}");
    }
    assert!(f.relative_error().iter().chain(&f.gain_error_db()).chain(&f.phase_error()).chain(&f.normalized_error()).all(|v| v.is_finite()));
    assert!(f.rms_relative_error().is_finite() && f.mean_normalized_error().is_finite() && c.mean_coherence().is_finite());
}

/// Durations in seconds: whole numbers of the sampling period, converted to samples; the
/// frequencies of the results in Hz.
#[test]
fn test_validation_durations_in_seconds() {
    use dsmc::system_identification::validation::{Validation, ValidationError};

    let ts = 1e-3;
    let plant = TransferFunction::continuous(&[1000.0], &[1.0, 20.0, 1000.0]);
    let u: Vec<f64> = (0..4000).map(|k| ((k * 7919) % 101) as f64 / 50.0 - 1.0).collect();
    let y: Vec<f64> = u.iter().enumerate().map(|(k, v)| 0.5 * v + 0.01 * ((k * 104729 % 997) as f64 / 498.5 - 1.0)).collect();
    let validation = Validation::continuous(&plant, ts, &u, &y).unwrap();

    let from = validation.clone().evaluated_from(1.5).unwrap();
    assert_eq!((from.start, from.samples()), (1500, 2500));
    let coherence = validation.coherence_test(0.5, 0.99).unwrap();
    assert_eq!(coherence.segment_len, 500);
    assert!(coherence.bins.iter().zip(coherence.frequencies()).all(|(&f, hz)| (hz - f as f64 * 2.0).abs() < 1e-9));
    let response = validation.frequency_response(0.25).unwrap();
    assert!(response.bins.iter().zip(response.frequencies()).all(|(&f, hz)| (hz - f as f64 * 4.0).abs() < 1e-9));
    let lines = validation.line_test(2.0, &[1, 3, 10], 0.99).unwrap(); // period 0.5 s
    assert_eq!(lines.periods, 8);
    assert_eq!(lines.frequencies(), vec![2.0, 6.0, 20.0]);

    let fractional = |duration: f64| ValidationError::FractionalDuration { duration, ts };
    assert_eq!(validation.clone().evaluated_from(1.0005).unwrap_err(), fractional(1.0005));
    assert_eq!(validation.coherence_test(0.5005, 0.99).unwrap_err(), fractional(0.5005));
    assert_eq!(validation.frequency_response(0.5005).unwrap_err(), fractional(0.5005));
    // 3 Hz at 1 kHz: 333.3 samples per period
    assert_eq!(validation.line_test(3.0, &[1], 0.99).unwrap_err(), ValidationError::FractionalPeriod { fundamental_frequency: 3.0, ts });
    assert!(matches!(validation.line_test(0.0, &[1], 0.99), Err(ValidationError::InvalidFundamental { .. })));
    assert!(matches!(validation.clone().evaluated_from(-1.0), Err(ValidationError::InvalidDuration { .. })));
    assert!(matches!(Validation::continuous(&plant, 0.0, &u, &y), Err(ValidationError::Simulation(dsmc::SimulationError::InvalidSamplingPeriod { .. }))));
    let g_z = TransferFunction::<f64, dsmc::Discrete>::discrete(&[1.0], &[1.0, -0.5]);
    assert!(matches!(Validation::discrete(&g_z, -1e-3, &u, &y), Err(ValidationError::InvalidDuration { .. })));
}
