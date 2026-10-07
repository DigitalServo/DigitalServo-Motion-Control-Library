use std::f64::consts::PI;

use dsmc::fft::welch;

const TS: f64 = 1e-4;

/// Multisine at the bins `bins` of a segment of `nperseg` samples (whole periods per segment).
fn multisine(n: usize, nperseg: usize, bins: impl Iterator<Item = usize>) -> Vec<f64> {
    let bins: Vec<usize> = bins.collect();
    (0..n)
        .map(|k| {
            bins.iter()
                .map(|&b| (2.0 * PI * (b * k) as f64 / nperseg as f64 + 0.1 * (b * b) as f64).cos())
                .sum::<f64>()
        })
        .collect()
}

/// Output of the discrete low-pass `y[k] = a y[k-1] + (1 - a) u[k-1]` plus a small noise.
fn plant(u: &[f64], noise: f64) -> Vec<f64> {
    let a = 0.95;
    let mut seed: u64 = 12345;
    let mut y = vec![0.0; u.len()];
    for k in 1..u.len() {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let r = (seed >> 11) as f64 / (1u64 << 53) as f64 - 0.5;
        y[k] = a * y[k - 1] + (1.0 - a) * u[k - 1];
        y[k] += noise * r;
    }
    y
}

#[test]
fn coherence_does_not_depend_on_the_units() {
    let (n, nperseg) = (16384, 1024);
    let u = multisine(n, nperseg, 1..200);
    let y = plant(&u, 1e-3);
    let (cu, cy) = (1e-4, 1e-5);
    let u_scaled: Vec<f64> = u.iter().map(|v| v * cu).collect();
    let y_scaled: Vec<f64> = y.iter().map(|v| v * cy).collect();

    let (g, gamma) = welch(&u, &y, TS, n / nperseg);
    let (g_scaled, gamma_scaled) = welch(&u_scaled, &y_scaled, TS, n / nperseg);

    assert_eq!(g.len(), nperseg / 2 + 1);
    for k in 1..200 {
        assert!(gamma[k] > 0.9, "bin {k}: {}", gamma[k]);
        assert!((gamma_scaled[k] - gamma[k]).abs() < 1e-9, "bin {k}: {} vs {}", gamma_scaled[k], gamma[k]);
        let expected = g[k].value * (cy / cu);
        assert!((g_scaled[k].value - expected).norm() < 1e-9 * expected.norm(), "bin {k}");
    }
}

#[test]
fn small_signals_keep_their_coherence() {
    let (n, nperseg) = (16384, 1024);
    let u: Vec<f64> = multisine(n, nperseg, 1..200).iter().map(|v| v * 1e-3).collect();
    let y: Vec<f64> = plant(&u, 0.0).iter().map(|v| v * 1e-3).collect();
    let y_max = y.iter().fold(0.0f64, |acc, v| acc.max(v.abs()));
    assert!(y_max < 1e-4, "{y_max}");

    let (g, gamma) = welch(&u, &y, TS, n / nperseg);
    for k in 1..200 {
        assert!(gamma[k] > 0.99, "bin {k}: {}", gamma[k]);
        assert!(g[k].value.norm() > 0.0, "bin {k}");
    }
}

#[test]
fn unexcited_bins_are_zero() {
    let (n, nperseg) = (16384, 1024);
    // The same band-limited input at the unit scale and at a small scale
    for scale in [1.0, 1e-6] {
        let u: Vec<f64> = multisine(n, nperseg, 10..50).iter().map(|v| v * scale).collect();
        let y = plant(&u, 0.0);
        let (g, gamma) = welch(&u, &y, TS, n / nperseg);
        for (k, c) in gamma.iter().enumerate().take(50).skip(10) {
            assert!(*c > 0.99, "scale {scale}, bin {k}: {c}");
        }
        for k in 200..=nperseg / 2 {
            assert_eq!(g[k].value.norm(), 0.0, "scale {scale}, bin {k}");
            assert_eq!(gamma[k], 0.0, "scale {scale}, bin {k}");
        }
    }
}

#[test]
fn zero_signals_give_zero() {
    let u = vec![0.0; 4096];
    let (g, gamma) = welch(&u, &u, TS, 4);
    assert!(g.iter().all(|r| r.value.norm() == 0.0));
    assert!(gamma.iter().all(|&c| c == 0.0));

    // Input excited, output identically zero: response zero, coherence zero (not NaN)
    let u = multisine(4096, 1024, 1..100);
    let y = vec![0.0; 4096];
    let (g, gamma) = welch(&u, &y, TS, 4);
    assert!(g.iter().all(|r| r.value.norm() == 0.0));
    assert!(gamma.iter().all(|&c| c == 0.0));
}

#[test]
fn segment_is_at_most_the_data() {
    // 2^ceil(log2(60000)) = 65536 > 60000: one segment of the whole data
    let n = 60000;
    let u = multisine(n, n, 1..100);
    let y = plant(&u, 0.0);
    let (g, gamma) = welch(&u, &y, TS, 1);
    assert_eq!(g.len(), n / 2 + 1);
    assert_eq!(gamma.len(), n / 2 + 1);
    assert!((g[1].omega - 2.0 * PI / (n as f64 * TS)).abs() < 1e-9);

    // More segments than samples, no segments, and too short data
    assert_eq!(welch(&u[..10], &y[..10], TS, 100).0.len(), 2);
    assert_eq!(welch(&u[..1000], &y[..1000], TS, 0).0.len(), 501);
    assert!(welch(&u[..1], &y[..1], TS, 1).0.is_empty());
    assert!(welch::<f64>(&[], &[], TS, 1).0.is_empty());
}

#[test]
#[should_panic(expected = "input and output lengths differ")]
fn lengths_must_match() {
    let u = vec![0.0; 1024];
    welch(&u, &u[..1000], TS, 1);
}
