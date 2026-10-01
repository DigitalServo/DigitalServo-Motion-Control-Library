use dsmc::{tf, StableInverseError};

fn assert_close(actual: f64, expected: f64, tol: f64) {
    assert!((actual - expected).abs() <= tol, "{actual} != {expected}");
}

/// Compare the two-sided `h(t)` with its analytic form on t ∈ [-5, 5].
fn check(h: impl Fn(f64) -> f64, analytic: impl Fn(f64) -> f64) {
    for i in -50..=50 {
        let t = 0.1 * i as f64;
        assert_close(h(t), analytic(t), 1e-9);
    }
}

#[test]
fn nonminimum_phase_first_order() {
    // G = (s - 1) / (s + 2)  ->  G^-1 = 1 + 3 / (s - 1)  ->  h = δ(t) - 3 e^t 1(-t)
    let inv = tf!("(s - 1) / (s + 2)").stable_inverse().unwrap();
    assert_eq!(inv.causal.direct.len(), 1);
    assert_close(inv.causal.direct[0], 1.0, 1e-12);
    assert!(inv.causal.terms.is_empty());
    check(inv.impulse_function(), |t| if t < 0.0 { -3.0 * t.exp() } else { 0.0 });
    println!("{}", inv.time_domain());
}

#[test]
fn mixed_stable_and_unstable_poles() {
    // G = (s - 1)(s + 2) / (s + 1)^3  ->  G^-1 = s + 2 + (8/3) / (s - 1) + (1/3) / (s + 2)
    let inv = tf!("(s - 1)(s + 2) / (s + 1)^3").stable_inverse().unwrap();
    assert_eq!(inv.causal.direct.len(), 2);
    assert_close(inv.causal.direct[0], 1.0, 1e-12);
    assert_close(inv.causal.direct[1], 2.0, 1e-12);
    let h = inv.impulse_function();
    check(&h, |t| if t < 0.0 { -8.0 / 3.0 * t.exp() } else { (-2.0 * t).exp() / 3.0 });

    // Bilateral transform at s = 0 (inside the ROC -2 < Re s < 1) of the strictly proper part:
    // ∫ h(t) dt = (3s + 5) / ((s - 1)(s + 2)) |_{s=0} = -5/2
    let dt = 1e-3;
    let integral: f64 = (-40_000..40_000).map(|i| h((i as f64 + 0.5) * dt) * dt).sum();
    assert_close(integral, -2.5, 1e-5);
    println!("{:.4}", inv.time_domain());
}

#[test]
fn unstable_complex_poles() {
    // G = (s^2 - 2s + 5) / (s + 1)^2  ->  G^-1 = 1 + 4(s - 1) / ((s - 1)^2 + 4)
    //   ->  h = δ(t) - 4 e^t cos(2t) 1(-t)
    let inv = tf!("(s^2 - 2s + 5) / (s + 1)^2").stable_inverse().unwrap();
    check(inv.impulse_function(), |t| if t < 0.0 { -4.0 * t.exp() * (2.0 * t).cos() } else { 0.0 });
    println!("{:.3}", inv.time_domain());
}

#[test]
fn minimum_phase_is_causal() {
    // G = (s + 3) / (s + 1)^2  ->  G^-1 = s - 1 + 4 / (s + 3): no anti-causal part
    let inv = tf!("(s + 3) / (s + 1)^2").stable_inverse().unwrap();
    assert!(inv.anticausal.terms.is_empty());
    check(inv.impulse_function(), |t| if t < 0.0 { 0.0 } else { 4.0 * (-3.0 * t).exp() });
}

#[test]
fn errors() {
    let err = tf!("(s^2 + 4) / (s + 1)^3").stable_inverse().unwrap_err();
    match err {
        StableInverseError::PoleOnImaginaryAxis { re, im } => {
            assert_eq!(re, 0.0);
            assert_close(im.abs(), 2.0, 1e-9);
        }
        e => panic!("unexpected error: {e}"),
    }
    assert!(matches!(
        tf!("s / (s + 1)").stable_inverse(),
        Err(StableInverseError::PoleOnImaginaryAxis { .. })
    ));
}

mod state_reference {
    use super::assert_close;
    use dsmc::discretize::exact_discretize::{DiscretizedSystem, LiftedDiscretizedSystem};
    use dsmc::{tf, trajectory, StableInverseError, TransferFunction};

    const TS: f64 = 1e-4;
    const REST: f64 = 0.02;
    const MOVE_SAMPLES: usize = 500;

    fn duration() -> f64 {
        (MOVE_SAMPLES - 1) as f64 * TS
    }

    #[test]
    fn sin_laplace_matches_generate() {
        let samples = trajectory::sin::generate(1.0, MOVE_SAMPLES);
        let y = trajectory::sin::laplace(1.0, duration(), REST).inverse_laplace();
        for (i, p) in samples.iter().enumerate() {
            assert_close(y(REST + i as f64 * TS), p.s, 1e-9);
        }
        assert_close(y(0.0), 0.0, 1e-12);
        assert_close(y(REST + duration() + 0.01), 1.0, 1e-9);
    }

    #[test]
    fn without_zeros_reference_is_trajectory() {
        // ξ = y, so the state reference is [s, v] (what tests/ptc.rs passes directly).
        let plant: TransferFunction<f64> = TransferFunction::continuous(&[1.0], &[2.0e-4, 0.05, 0.0]);
        let reference = plant.state_reference(&trajectory::sin::laplace(1.0, duration(), REST)).unwrap();
        assert_eq!(reference.order(), 2);
        let samples = trajectory::sin::generate(1.0, MOVE_SAMPLES);
        for (i, p) in samples.iter().enumerate() {
            let x = reference.state(REST + i as f64 * TS);
            assert_close(x[0], p.s, 1e-9);
            assert_close(x[1], p.v / duration(), 1e-6);
        }
    }

    #[test]
    fn nonminimum_phase_perfect_tracking() {
        // Zero at s = +1000 [rad/s]: y = ξ - ξ' / 1000
        let plant = tf!("(1 - 0.001s) / (0.0002s^2 + 0.05s)");
        let y_d = trajectory::sin::laplace(1.0, duration(), REST);
        let reference = plant.state_reference(&y_d).unwrap();
        let y = y_d.inverse_laplace();

        // Output equation holds at all times, and pre-actuation starts before the move.
        for i in -100..=1000 {
            let t = REST + i as f64 * TS;
            let x = reference.state(t);
            assert_close(x[0] - x[1] / 1000.0, y(t), 1e-9);
        }
        assert!(reference.state(REST - 10.0 * TS)[0].abs() > 1e-6);
        assert!(reference.state(0.0)[0].abs() < 1e-8);

        // Perfect tracking control on the lifted model: y matches y_d at every frame.
        let total = ((2.0 * REST + duration()) / TS).round() as usize;
        let r = reference.sample(0.0, TS, total);
        let mut model = DiscretizedSystem::from_tf(&plant, TS).unwrap();
        let lifted: LiftedDiscretizedSystem<f64> = model.clone().try_into().unwrap();
        let u = lifted.calculate_ptc_input_from_reference_state(r);
        let mut max_error: f64 = 0.0;
        for (i, &ui) in u.iter().enumerate() {
            model.update(&[ui]).unwrap();
            if (i + 1) % reference.order() == 0 {
                max_error = max_error.max((model.output[0] - y((i + 1) as f64 * TS)).abs());
            }
        }
        assert!(max_error < 1e-8, "max tracking error at frames: {max_error:e}");
    }

    #[test]
    fn errors() {
        let y_d = trajectory::sin::laplace(1.0, duration(), REST);
        // Relative degree 4: ξ''' needs y''' of the sin profile, which has impulses.
        assert!(matches!(
            tf!("1 / (s + 1)^4").state_reference(&y_d),
            Err(StableInverseError::NotSmoothEnough { state: 3 })
        ));
        // Zero on the imaginary axis
        assert!(matches!(
            tf!("(s^2 + 4) / (s + 1)^3").state_reference(&y_d),
            Err(StableInverseError::PoleOnImaginaryAxis { .. })
        ));
    }
}

mod polynomial_reference {
    use super::assert_close;
    use dsmc::discretize::exact_discretize::{DiscretizedSystem, LiftedDiscretizedSystem};
    use dsmc::{tf, trajectory, ReferenceSignal, StableInverseError};

    const TS: f64 = 1e-4;
    const REST: f64 = 0.02;
    const DURATION: f64 = 0.05;

    #[test]
    fn normalized_coefficients() {
        let cubic: Vec<f64> = trajectory::polynomial::normalized_coefficients(1);
        assert_eq!(cubic, vec![0.0, 0.0, 3.0, -2.0]);
        let quintic: Vec<f64> = trajectory::polynomial::normalized_coefficients(2);
        assert_eq!(quintic, vec![0.0, 0.0, 0.0, 10.0, -15.0, 6.0]);
    }

    #[test]
    fn smoothness_at_both_ends() {
        for k in 0..=8 {
            let y = trajectory::polynomial::piecewise(2.0, DURATION, REST, k);
            let before = y.derivatives(REST, k + 2);
            let after = y.derivatives(REST + DURATION, k + 2);
            assert_close(after[0], 2.0, 1e-9);
            for m in 1..=k {
                // m-th derivative ~ distance / duration^m: compare relative to that scale
                let scale = 2.0 / DURATION.powi(m as i32);
                assert_close(before[m] / scale, 0.0, 1e-9);
                assert_close(after[m] / scale, 0.0, 1e-9);
            }
            assert!(before[k + 1].abs() > 0.0, "y^(k+1) should jump (k = {k})");
        }
    }

    #[test]
    fn without_zeros_reference_is_derivatives() {
        // ξ = y: the state reference is [y, y', y'', y'''] of the trajectory itself.
        let plant = tf!("1 / (s (0.005s + 1) (0.001s + 1) (0.0005s + 1))");
        let y_d = trajectory::polynomial::piecewise(1.0, DURATION, REST, 3);
        let reference = plant.state_reference(&y_d).unwrap();
        for i in 0..1000 {
            let t = i as f64 * TS;
            let x = reference.state(t);
            let d = y_d.derivatives(t, 4);
            for m in 0..4 {
                assert_close(x[m] / DURATION.powi(-(m as i32)), d[m] / DURATION.powi(-(m as i32)), 1e-9);
            }
        }
    }

    /// Relative degree 4 with a zero at +1000 rad/s (beyond the sin profile, which allows ρ <= 3).
    fn plant() -> dsmc::TransferFunction<f64> {
        tf!("(1 - 0.001s) / (s (0.005s + 1) (0.001s + 1) (0.0005s + 1) (0.0002s + 1))")
    }

    /// `y = N(s) / N(0) ξ = ξ - ξ' / 1000` must hold at all times, including long after the move.
    #[test]
    fn output_equation_holds_long_after_the_move() {
        let y_d = trajectory::polynomial::piecewise(1.0, DURATION, REST, 4);
        let reference = plant().state_reference(&y_d).unwrap();
        for t in [0.0, REST - 0.002, REST, REST + 0.01, REST + DURATION, 0.2, 1.0, 10.0, 100.0] {
            let x = reference.state(t);
            assert_close(x[0] - x[1] / 1000.0, y_d.value(t), 1e-9);
        }
        // At rest at the end: ξ = distance, all derivatives zero.
        let x = reference.state(100.0);
        assert_close(x[0], 1.0, 1e-12);
        for &xj in &x[1..] {
            assert!(xj.abs() < 1e-9, "{x:?}");
        }
        // Pre-actuation before the move (anti-causal part of the zero at +1000).
        assert!(reference.state(REST - 0.001)[0].abs() > 1e-9);
    }

    /// The jump decomposition through `LaplaceSignal` gives the same reference near the move.
    #[test]
    fn matches_laplace_signal_path_near_the_move() {
        let y_d = trajectory::polynomial::piecewise(1.0, DURATION, REST, 4);
        let exact = plant().state_reference(&y_d).unwrap();
        let laplace = plant().state_reference(&y_d.laplace()).unwrap();
        for i in 0..800 {
            let t = i as f64 * TS;
            for (m, (a, b)) in exact.state(t).iter().zip(laplace.state(t)).enumerate() {
                let scale = DURATION.powi(m as i32);
                assert_close(a * scale, b * scale, 1e-6);
            }
        }
    }

    #[test]
    fn perfect_tracking() {
        let plant = plant();
        let y_d = trajectory::polynomial::piecewise(1.0, DURATION, REST, 4);
        let reference = plant.state_reference(&y_d).unwrap();
        let total = ((2.0 * REST + DURATION) / TS).round() as usize;
        let r = reference.sample(0.0, TS, total);
        let mut model = DiscretizedSystem::from_tf(&plant, TS).unwrap();
        let lifted: LiftedDiscretizedSystem<f64> = model.clone().try_into().unwrap();
        let u = lifted.calculate_ptc_input_from_reference_state(r);
        let mut max_error: f64 = 0.0;
        for (i, &ui) in u.iter().enumerate() {
            model.update(&[ui]).unwrap();
            if (i + 1) % reference.order() == 0 {
                max_error = max_error.max((model.output[0] - y_d.value((i + 1) as f64 * TS)).abs());
            }
        }
        assert!(max_error < 1e-6, "max tracking error at frames: {max_error:e}");
    }

    #[test]
    fn not_smooth_enough() {
        // ρ = 4 needs k >= 2: with k = 1, ξ'''' would contain impulses (state index 1 + 2 + 1 = 4).
        let y_d = trajectory::polynomial::piecewise(1.0, DURATION, REST, 1);
        assert!(matches!(
            plant().state_reference(&y_d),
            Err(StableInverseError::NotSmoothEnough { state: 4 })
        ));
        let y_d = trajectory::polynomial::piecewise(1.0, DURATION, REST, 2);
        assert!(plant().state_reference(&y_d).is_ok());
        let _: &dyn ReferenceSignal<f64> = &y_d;
    }

    #[test]
    fn generate_matches_piecewise() {
        let samples = trajectory::polynomial::generate(1.0, 101, 3);
        let y = trajectory::polynomial::piecewise(1.0, 1.0, 0.0, 3);
        for (i, p) in samples.iter().enumerate() {
            let d = y.derivatives(i as f64 / 100.0, 3);
            assert_close(p.s, d[0], 1e-12);
            assert_close(p.v, d[1], 1e-9);
            assert_close(p.a, d[2], 1e-9);
        }
        assert_close(samples[100].s, 1.0, 1e-12);
    }
}
