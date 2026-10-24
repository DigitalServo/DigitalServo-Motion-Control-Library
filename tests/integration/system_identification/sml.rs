//! Frequency-domain sample maximum likelihood (SML) estimator.

use dsmc::signal::excitation::multisine;
use dsmc::system_identification::frequency_response::levy;
use dsmc::system_identification::frequency_response::sml::{self, SmlData, SmlError, SmlOptions, SpectralLine};
use dsmc::{FrequencyResponse, TransferFunction};
use num_complex::Complex64;

use super::*;

/// Standard normal pseudo-random numbers (xorshift64, Box-Muller).
fn gaussian_source(mut state: u64) -> impl FnMut() -> f64 {
    move || {
        let mut uniform = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            ((state >> 11) as f64 + 0.5) / (1u64 << 53) as f64
        };
        (-2.0 * uniform().ln()).sqrt() * (2.0 * std::f64::consts::PI * uniform()).cos()
    }
}

const TS: f64 = 1e-3;
const PERIOD: usize = 256;

/// Lines 1 ..= 100 of the period of 256 samples.
fn excited_lines() -> Vec<usize> {
    (1..=100).collect()
}

/// `y0[k] = 1.5 y0[k-1] - 0.7 y0[k-2] + u0[k-1] + 0.5 u0[k-2]`, i.e.
/// `G(z) = (z + 0.5) / (z^2 - 1.5 z + 0.7)` (n = 2, m = 1, nk = 1), driven by a multisine of
/// `periods + 1` periods; the first (transient) period is left out. The measured input and output
/// carry white Gaussian noise of `noise_u`, `noise_y` times the RMS of `u0`, `y0`
/// (errors-in-variables).
fn measured(periods: usize, (noise_u, noise_y): (f64, f64), gaussian: &mut impl FnMut() -> f64) -> (Vec<f64>, Vec<f64>) {
    let total = (periods + 1) * PERIOD;
    let f0 = 1.0 / (PERIOD as f64 * TS);
    let u0: Vec<f64> = multisine(total as f64 * TS, TS, f0, &excited_lines(), |_| 1.0, 3).unwrap();
    let mut y0 = vec![0.0; total];
    for k in 0..total {
        let y = |i: usize| if k >= i { y0[k - i] } else { 0.0 };
        let u = |i: usize| if k >= i { u0[k - i] } else { 0.0 };
        y0[k] = 1.5 * y(1) - 0.7 * y(2) + u(1) + 0.5 * u(2);
    }
    let rms = |x: &[f64]| (x.iter().map(|v| v * v).sum::<f64>() / x.len() as f64).sqrt();
    let (su, sy) = (noise_u * rms(&u0), noise_y * rms(&y0));
    let u: Vec<f64> = u0[PERIOD..].iter().map(|v| v + su * gaussian()).collect();
    let y: Vec<f64> = y0[PERIOD..].iter().map(|v| v + sy * gaussian()).collect();
    (u, y)
}

fn discrete_plant() -> TransferFunction<f64, dsmc::Discrete> {
    TransferFunction::from_polynomials(dsmc::Polynomial(vec![0.0, 1.0, 0.5]), dsmc::Polynomial(vec![1.0, -1.5, 0.7]))
}

/// Noise on the input (5 %) and the output (20 %), 8 periods: the discrete-time model is found
/// within a few standard deviations of each coefficient, and the cost is within its expected
/// spread (no model error).
#[test]
fn test_sml_discrete_errors_in_variables() {
    let mut gaussian = gaussian_source(0x2545_f491_4f6c_dd1d);
    let (u, y) = measured(8, (0.05, 0.2), &mut gaussian);
    let data = SmlData::from_periods(&u, &y, TS, PERIOD, &excited_lines()).unwrap();
    let result = sml::identify_discrete(&data, TS, 2, 1, 1, &SmlOptions::default()).unwrap();
    println!("{:?}\nstd {:?} {:?}\ncost {:.1}, expected {:.1} ± {:.1}, {} iterations", result.model, result.std_numerator, result.std_denominator, result.cost, result.expected_cost, result.cost_std, result.iterations);
    assert!(result.converged(), "{:?}", result.termination);

    let expected = discrete_plant();
    assert_eq!(result.model.numerator.len(), 3);
    assert_eq!(result.model.denominator.len(), 3);
    let coefficients = result.model.numerator.iter().zip(&expected.numerator.0).zip(&result.std_numerator);
    let coefficients = coefficients.chain(result.model.denominator.iter().zip(&expected.denominator.0).zip(&result.std_denominator));
    for ((&estimate, &truth), &std) in coefficients {
        assert!((estimate - truth).abs() <= 4.0 * std + 1e-12, "{estimate} vs {truth} (std {std:e})");
        assert!((estimate - truth).abs() < 0.02, "{estimate} vs {truth}");
    }
    assert!((result.cost - result.expected_cost).abs() < 4.0 * result.cost_std, "cost {} vs {} ± {}", result.cost, result.expected_cost, result.cost_std);
    assert!(tf_distance(&result.model, &expected) < 0.02);
}

/// Monte Carlo over 40 noise realizations: the estimate is unbiased (mean within 3 standard errors
/// of the truth), the reported standard deviations match the spread of the estimates, and the mean
/// cost its expected value.
#[test]
fn test_sml_discrete_monte_carlo() {
    let mut gaussian = gaussian_source(0x9e37_79b9_7f4a_7c15);
    let runs = 40;
    let truth = [1.0, 0.5, -1.5, 0.7];
    let mut estimates: Vec<[f64; 4]> = Vec::new();
    let mut reported = [0.0; 4];
    let (mut cost, mut expected, mut cost_std) = (0.0, 0.0, 0.0);
    for _ in 0..runs {
        let (u, y) = measured(6, (0.05, 0.3), &mut gaussian);
        let data = SmlData::from_periods(&u, &y, TS, PERIOD, &excited_lines()).unwrap();
        let result = sml::identify_discrete(&data, TS, 2, 1, 1, &SmlOptions::default()).unwrap();
        cost += result.cost / runs as f64;
        (expected, cost_std) = (result.expected_cost, result.cost_std);
        let (b, a) = (&result.model.numerator, &result.model.denominator);
        estimates.push([b[1], b[2], a[1], a[2]]);
        let std = [result.std_numerator[1], result.std_numerator[2], result.std_denominator[1], result.std_denominator[2]];
        for i in 0..4 {
            reported[i] += std[i] / runs as f64;
        }
    }
    for i in 0..4 {
        let mean = estimates.iter().map(|e| e[i]).sum::<f64>() / runs as f64;
        let spread = (estimates.iter().map(|e| (e[i] - mean).powi(2)).sum::<f64>() / (runs - 1) as f64).sqrt();
        println!("θ{i}: mean {mean:.5} (truth {}), spread {spread:.2e}, reported {:.2e}", truth[i], reported[i]);
        assert!((mean - truth[i]).abs() < 3.0 * spread / (runs as f64).sqrt(), "biased: {mean} vs {}", truth[i]);
        assert!(spread / reported[i] > 0.6 && spread / reported[i] < 1.6, "spread {spread:e} vs reported {:e}", reported[i]);
    }
    println!("mean cost {cost:.1}, expected {expected:.1} ± {:.1}", cost_std / (runs as f64).sqrt());
    assert!((cost - expected).abs() < 3.0 * cost_std / (runs as f64).sqrt());
}

/// A model of too low an order leaves a cost far above its expected value.
#[test]
fn test_sml_cost_detects_model_error() {
    let mut gaussian = gaussian_source(0x2545_f491_4f6c_dd1d);
    let (u, y) = measured(8, (0.05, 0.2), &mut gaussian);
    let data = SmlData::from_periods(&u, &y, TS, PERIOD, &excited_lines()).unwrap();
    let result = sml::identify_discrete(&data, TS, 1, 1, 1, &SmlOptions::default()).unwrap();
    println!("cost {:.1}, expected {:.1} ± {:.1}", result.cost, result.expected_cost, result.cost_std);
    assert!(result.cost > result.expected_cost + 10.0 * result.cost_std);
}

/// Lines of `g` at `omegas` from `periods` periods of the spectra `U = 1 + n_u`, `Y = G + n_y` with
/// circular complex Gaussian noise of standard deviations `noise_u`, `noise_y |G|` (the same
/// relative noise at every line), and the sample means and (co)variances of the means.
fn synthetic_lines(g: &TransferFunction<f64>, omegas: &[f64], periods: usize, (noise_u, noise_y): (f64, f64), gaussian: &mut impl FnMut() -> f64) -> Vec<SpectralLine<f64>> {
    let response = g.frequency_transfer_function();
    let m = periods as f64;
    let mut complex = |std: f64| Complex64::new(gaussian(), gaussian()) * (std / 2f64.sqrt());
    omegas
        .iter()
        .map(|&omega| {
            let g0 = response.response(omega);
            let us: Vec<Complex64> = (0..periods).map(|_| 1.0 + complex(noise_u)).collect();
            let ys: Vec<Complex64> = (0..periods).map(|_| g0 + complex(noise_y * g0.norm())).collect();
            let (u, y) = (us.iter().sum::<Complex64>() / m, ys.iter().sum::<Complex64>() / m);
            let scale = 1.0 / (m * (m - 1.0));
            SpectralLine {
                omega,
                u,
                y,
                var_u: us.iter().map(|v| (v - u).norm_sqr()).sum::<f64>() * scale,
                var_y: ys.iter().map(|v| (v - y).norm_sqr()).sum::<f64>() * scale,
                cov_yu: us.iter().zip(&ys).map(|(a, b)| (b - y) * (a - u).conj()).sum::<Complex64>() * scale,
            }
        })
        .collect()
}

/// `100 / ((s + 1) (s + 100))` over four decades (0.1 .. 1000 rad/s) with 5 % noise relative to
/// `|G|` at every line and 2 % on the input: the SML model (continuous time, scaled `s`) finds both
/// poles within a few percent, while Sanathanan-Koerner on the mean response `Ȳ / Ū` (absolute
/// error) misses the fast one by more.
#[test]
fn test_sml_continuous_wide_band() {
    let mut gaussian = gaussian_source(0x2545_f491_4f6c_dd1d);
    let g = TransferFunction::continuous(&[100.0], &[1.0, 101.0, 100.0]);
    let omegas: Vec<f64> = (0..80).map(|k| 0.1 * 10f64.powf(4.0 * k as f64 / 79.0)).collect();
    let data = SmlData::new(synthetic_lines(&g, &omegas, 6, (0.02, 0.05), &mut gaussian), 6).unwrap();
    let result = sml::identify_continuous(&data, 2, 0, &SmlOptions::default()).unwrap();
    println!("{:?}\nstd {:?} {:?}\ncost {:.1}, expected {:.1} ± {:.1}", result.model, result.std_numerator, result.std_denominator, result.cost, result.expected_cost, result.cost_std);
    assert!(result.converged(), "{:?}", result.termination);
    assert!((result.cost - result.expected_cost).abs() < 4.0 * result.cost_std);

    let fast_pole = |tf: &TransferFunction<f64>| tf.pz_map().poles.iter().map(|p| p.norm()).fold(0.0f64, f64::max);
    let slow_pole = |tf: &TransferFunction<f64>| tf.pz_map().poles.iter().map(|p| p.norm()).fold(f64::INFINITY, f64::min);
    let samples: Vec<FrequencyResponse<f64>> = data.lines.iter().map(|l| FrequencyResponse { omega: l.omega, value: l.y / l.u }).collect();
    let sk: TransferFunction<f64> = levy::sanathanan_koerner_identification(&samples, 2, 0, &levy::SanathananKoernerOptions::default()).unwrap().into();
    let error = |p: f64, truth: f64| (p / truth - 1.0).abs();
    println!("fast pole: SML {:.3}, SK {:.3}; slow pole: SML {:.4}, SK {:.4}", fast_pole(&result.model), fast_pole(&sk), slow_pole(&result.model), slow_pole(&sk));
    assert!(error(fast_pole(&result.model), 100.0) < 0.03 && error(slow_pole(&result.model), 1.0) < 0.03);
    assert!(error(fast_pole(&result.model), 100.0) < error(fast_pole(&sk), 100.0));
    let k = result.model.denominator[0];
    assert!(error(result.model.numerator[0] / k, 100.0) < 0.03, "{:?}", result.model);
}

/// A lightly damped resonance `1000 / (s^2 + 2 s + 1000)` (ζ ≈ 0.03) with 5 % noise: the damping
/// is found within a few standard deviations.
#[test]
fn test_sml_continuous_resonance() {
    let mut gaussian = gaussian_source(0x9e37_79b9_7f4a_7c15);
    let g = TransferFunction::continuous(&[1000.0], &[1.0, 2.0, 1000.0]);
    let omegas: Vec<f64> = (1..=120).map(|k| 0.5 * k as f64).collect();
    let data = SmlData::new(synthetic_lines(&g, &omegas, 8, (0.0, 0.05), &mut gaussian), 8).unwrap();
    let result = sml::identify_continuous(&data, 2, 0, &SmlOptions::default()).unwrap();
    println!("{:?}\nstd {:?}", result.model, result.std_denominator);
    assert!(result.converged());
    let (a, std) = (&result.model.denominator, &result.std_denominator);
    assert!((a[1] - 2.0).abs() < 4.0 * std[1] && (a[2] - 1000.0).abs() < 4.0 * std[2], "{a:?} ± {std:?}");
    assert!((a[1] / 2.0 - 1.0).abs() < 0.05);
}

#[test]
fn test_sml_errors() {
    let lines = excited_lines();
    let u = vec![1.0; 3 * PERIOD];
    assert_eq!(SmlData::from_periods(&u, &u[1..], TS, PERIOD, &lines).unwrap_err(), SmlError::LengthMismatch { input: 3 * PERIOD, output: 3 * PERIOD - 1 });
    assert_eq!(SmlData::from_periods(&u[1..], &u[1..], TS, PERIOD, &lines).unwrap_err(), SmlError::NotWholePeriods { samples: 3 * PERIOD - 1, period: PERIOD });
    assert_eq!(SmlData::from_periods(&u, &u, TS, PERIOD, &lines).unwrap_err(), SmlError::TooFewPeriods { periods: 3 });
    let u = vec![1.0; 4 * PERIOD];
    assert_eq!(SmlData::from_periods(&u, &u, TS, PERIOD, &[129]).unwrap_err(), SmlError::InvalidLine { line: 129, period: PERIOD });
    // Noise-free data: no variance to weight with
    assert_eq!(SmlData::from_periods(&u, &u, TS, PERIOD, &[1]).unwrap_err(), SmlError::InvalidSpectralLine { index: 0 });

    let mut gaussian = gaussian_source(1);
    let (u, y) = measured(4, (0.05, 0.2), &mut gaussian);
    let data = SmlData::from_periods(&u, &y, TS, PERIOD, &[1, 2]).unwrap();
    assert_eq!(sml::identify_discrete(&data, TS, 2, 1, 1, &SmlOptions::default()).unwrap_err(), SmlError::TooFewLines { lines: 2, parameters: 4 });
    assert_eq!(sml::identify_discrete(&data, 0.0, 1, 0, 0, &SmlOptions::default()).unwrap_err(), SmlError::InvalidSamplingPeriod);
}
