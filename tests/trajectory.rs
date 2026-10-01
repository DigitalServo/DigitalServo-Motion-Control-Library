use dsmc::logger::DataStorage;
use dsmc::trajectory::{Trajectory, TrajectoryKind};

#[test]
fn test_trajectory() {

    let samples = 800;
    let distance = 10.0;

    let trajectory_sin = TrajectoryKind::Sin.generate(distance, samples);
    let trajectory_cycloid = TrajectoryKind::Cycloid.generate(distance, samples);
    let trajectory_mt = TrajectoryKind::ModifiedTrapezoid.generate(distance, samples);
    let trajectory_ms = TrajectoryKind::ModifiedSine.generate(distance, samples);
    let trajectory_mcv20 = TrajectoryKind::ModifiedConstantVelocity { constant_velocity_percent: 50.0 }.generate(distance, samples);
    let trajectory_mcv80 = TrajectoryKind::ModifiedConstantVelocity { constant_velocity_percent: 80.0 }.generate(distance, samples);
    let trajectory_poly = TrajectoryKind::SmoothPolynomial { smoothness: 4 }.generate(distance, samples);

    let mut storage = DataStorage::new("./out/trajectory.csv", ',', false).unwrap();

    for i in 0..samples {
        storage.add(&[
            i as f64,
            trajectory_ms[i].s,
            trajectory_mt[i].s,
            trajectory_mcv20[i].s,
            trajectory_sin[i].s,
            trajectory_cycloid[i].s,
            trajectory_mcv80[i].s,
            trajectory_poly[i].s
        ]).unwrap();
    }

    storage.close().unwrap();
}

mod traits {
    use dsmc::discretize::exact_discretize::DiscretizedSystem;
    use dsmc::feedforward::ptc::{LiftedDiscretizedSystem, ReferenceSignal};
    use dsmc::tf;
    use dsmc::trajectory::{
        Cycloid, ModifiedConstantVelocity, ModifiedSine, ModifiedTrapezoid, SignalTrajectory, Sin, SmoothPolynomial,
        Trajectory, TrajectoryKind,
    };

    fn all() -> [TrajectoryKind; 6] {
        [
            TrajectoryKind::Sin,
            TrajectoryKind::Cycloid,
            TrajectoryKind::ModifiedTrapezoid,
            TrajectoryKind::ModifiedSine,
            TrajectoryKind::ModifiedConstantVelocity { constant_velocity_percent: 40.0 },
            TrajectoryKind::SmoothPolynomial { smoothness: 3 },
        ]
    }

    #[test]
    fn kind_matches_individual_types() {
        let individual: [(TrajectoryKind, &dyn Trajectory<f64>); 6] = [
            (Sin.into(), &Sin),
            (Cycloid.into(), &Cycloid),
            (ModifiedTrapezoid.into(), &ModifiedTrapezoid),
            (ModifiedSine.into(), &ModifiedSine),
            (ModifiedConstantVelocity { constant_velocity_percent: 40.0 }.into(), &ModifiedConstantVelocity { constant_velocity_percent: 40.0 }),
            (SmoothPolynomial { smoothness: 3 }.into(), &SmoothPolynomial { smoothness: 3 }),
        ];
        assert_eq!(individual.map(|(kind, _)| kind), all());
        for (kind, p) in individual {
            for i in 0..=20 {
                let x = i as f64 / 20.0;
                assert_eq!(kind.profile(2.0, x).s, p.profile(2.0, x).s, "{kind:?} at {x}");
            }
        }
    }

    #[test]
    fn rest_to_rest_with_consistent_derivatives() {
        let distance: f64 = 2.0;
        for p in all() {
            let name = format!("{p:?}");
            assert_eq!(p.profile(distance, -0.1).s, 0.0, "{name}");
            let end = p.profile(distance, 1.5);
            assert_eq!((end.s, end.v, end.a), (distance, 0.0, 0.0), "{name}");
            assert!((p.profile(distance, 1.0).s - distance).abs() < 1e-12, "{name}");

            // v and a are the x-derivatives of s and v (central differences).
            let h = 1e-6;
            for i in 1..100 {
                let x = i as f64 / 100.0;
                let (lo, mid, hi) = (p.profile(distance, x - h), p.profile(distance, x), p.profile(distance, x + h));
                assert!(((hi.s - lo.s) / (2.0 * h) - mid.v).abs() < 1e-4, "{name} v at {x}");
                assert!(((hi.v - lo.v) / (2.0 * h) - mid.a).abs() < 1e-4, "{name} a at {x}");
            }

            let samples = p.generate(distance, 101);
            assert_eq!(samples.len(), 101);
            assert_eq!(samples[50].s, p.profile(distance, 0.5).s, "{name}");
        }
    }

    /// The exact signal on the time axis agrees with the normalized profile.
    #[test]
    fn signal_matches_profile() {
        let (distance, duration, start) = (1.5, 0.05, 0.02);
        let sin = Sin.signal(distance, duration, start).inverse_laplace();
        let poly = SmoothPolynomial { smoothness: 4 }.signal(distance, duration, start);
        for i in 0..=100 {
            let t = i as f64 * 1e-3;
            let x = (t - start) / duration;
            assert!((sin(t) - Sin.profile(distance, x).s).abs() < 1e-9, "sin at {t}");
            let expected = SmoothPolynomial { smoothness: 4 }.profile(distance, x).s;
            assert!((poly.value(t) - expected).abs() < 1e-9, "poly at {t}");
        }
    }

    /// Any `SignalTrajectory` whose signal is a `ReferenceSignal` can drive output-reference PTC.
    fn ptc_error<P: Trajectory<f64> + SignalTrajectory<f64>>(profile: &P) -> f64
    where
        P::Signal: ReferenceSignal<f64>,
    {
        let (ts, rest, duration) = (1e-4, 0.02, 0.05);
        let plant = tf!("(1 - 0.001s) / (0.0002s^2 + 0.05s)");
        let mut model = DiscretizedSystem::from_tf(&plant, ts).unwrap();
        let lifted: LiftedDiscretizedSystem<f64> = model.clone().try_into().unwrap();
        let samples = ((2.0 * rest + duration) / ts).round() as usize;
        let u = lifted
            .calculate_ptc_input_for_reference_output(&profile.signal(1.0, duration, rest), 0.0, samples)
            .unwrap();
        let mut max_error: f64 = 0.0;
        for (i, &ui) in u.iter().enumerate() {
            model.update(&[ui]).unwrap();
            if (i + 1) % 2 == 0 {
                let x = ((i + 1) as f64 * ts - rest) / duration;
                max_error = max_error.max((model.output[0] - profile.profile(1.0, x).s).abs());
            }
        }
        max_error
    }

    #[test]
    fn signal_trajectories_drive_ptc() {
        assert!(ptc_error(&Sin) < 1e-8);
        assert!(ptc_error(&SmoothPolynomial { smoothness: 2 }) < 1e-8);
    }
}
