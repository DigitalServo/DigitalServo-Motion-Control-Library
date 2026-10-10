//! Inertia `J` and viscous friction `D` of the rigid-body mode of a two-inertia system with a high
//! resonance, from the torque `τ` and the measured load position `θ`, by least squares on
//! `τ = J θ'' + D θ'` (`lsm::whitebox`), all the signals filtered by the same state-variable filter
//! `F(s) = λ^3 / (s + λ)^3`:
//!
//! ```text
//! τ_f = F(s) τ,   v_f = s F(s) θ,   a_f = s^2 F(s) θ,   τ_f = J a_f + D v_f
//! ```
//!
//! The same filter on both sides keeps the relation, and with `λ` below the resonance it removes
//! the resonance and the noise of the measured position, amplified by the derivatives (which
//! would otherwise bias `J` down, as noise in the regressors). `F` is of the third order so that
//! `s^2 F` is strictly proper: it still rolls off the noise above `λ`. The derivatives of `F` also
//! leave out the constant offset of the position from rest. The torque is held between samples
//! and the position is a sampled continuous signal: the filters are discretized by `Zoh` for `τ`
//! and by `Tustin` for `θ`, so that they are not shifted by half a sample against each other (a
//! phase error `ω ts / 2` that `J ω` would turn into an error of `D`).
//!
//! Prints the estimates against `λ` (also `out/whitebox_rigid_body_lambda.csv`), then the
//! coherence test of the rigid-body model on the whole band and below the resonance, and its gain
//! and phase errors against the data. The signals are written to `out/whitebox_rigid_body.csv`
//! (`t, τ, θ, θ without noise`).
//!
//! `cargo run --release --example whitebox_rigid_body`

use std::f64::consts::PI;

use dsmc::logger::DataStorage;
use dsmc::signal::excitation::multisine;
use dsmc::system_identification::lsm::whitebox::DataBuffer;
use dsmc::system_identification::validation::{Check, CoherenceCheck, Validation};
use dsmc::{Discrete, DiscreteSystem, TransferFunction, discretize::{Tustin, Zoh}, tf};

const TS: f64 = 1e-4;

/// `gz` applied to `x` from rest.
fn filter(gz: &TransferFunction<f64, Discrete>, x: &[f64]) -> Vec<f64> {
    let mut filter = DiscreteSystem::try_from(gz).unwrap();
    x.iter().map(|&v| filter.update(v)).collect()
}

/// Least-squares `[J, D]` of `y = J a + D v` from sample `from` on.
fn estimate(a: &[f64], v: &[f64], y: &[f64], from: usize) -> [f64; 2] {
    let mut buffer = DataBuffer::new(2);
    for k in from..y.len() {
        buffer.add(&[a[k], v[k]], y[k]);
    }
    let theta = buffer.identify().unwrap();
    [theta[0], theta[1]]
}

fn main() {
    // Two-inertia system, torque to load position, without antiresonance:
    // θ / τ = 1 / (s (J s + D)) · ωr^2 / (s^2 + 2 ζ ωr s + ωr^2)
    let (j, d) = (0.01, 0.05); // [kg m^2], [N m s / rad]
    let (fr, zeta) = (300.0, 0.03); // resonance [Hz]
    let wr = 2.0 * PI * fr;
    let rigid = TransferFunction::continuous(&[1.0], &[j, d, 0.0]);
    let plant = &rigid * &TransferFunction::continuous(&[wr * wr], &[1.0, 2.0 * zeta * wr, wr * wr]);

    // Torque: multisine 1 ..= 50 Hz, 1 N m RMS, 10 s (its phases keep the position from drifting).
    // Position noise 1e-5 rad (white)
    let tlen = 10.0;
    let lines: Vec<usize> = (1..=50).collect();
    let tau: Vec<f64> = multisine(tlen, TS, 1.0, &lines, |_| 1.0, 1).unwrap();
    let theta0 = filter(&plant.discretize(Zoh, TS).unwrap(), &tau);
    let mut state: u64 = 0x2545_f491_4f6c_dd1d;
    let mut uniform = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        ((state >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    };
    let mut gaussian = move || (-2.0 * uniform().ln()).sqrt() * (2.0 * PI * uniform()).cos();
    let theta: Vec<f64> = theta0.iter().map(|x| x + 1e-5 * gaussian()).collect();

    // Evaluated after the transient from rest (J / D = 0.2 s)
    let from = (2.0 / TS) as usize;
    let error = |[jh, dh]: [f64; 2]| (100.0 * (jh / j - 1.0), 100.0 * (dh / d - 1.0));
    println!("true: J = {j:.5} kg m^2, D = {d:.5} N m s/rad, resonance {fr} Hz, excited 1 ..= 50 Hz");

    // State-variable filter of bandwidth `lambda_hz` [Hz]: `[J, D]`
    let identify = |lambda_hz: f64| {
        let l = 2.0 * PI * lambda_hz;
        let f = tf!("({l} / (s + {l}))^3");
        let (fs1, fs2) = (tf!("s") * f.clone(), tf!("s^2") * f.clone());
        let (f_z, fs1_z, fs2_z) = (f.discretize(Zoh, TS).unwrap(), fs1.discretize(Tustin, TS).unwrap(), fs2.discretize(Tustin, TS).unwrap());
        let (tau_f, v_f, a_f) = (filter(&f_z, &tau), filter(&fs1_z, &theta), filter(&fs2_z, &theta));
        estimate(&a_f, &v_f, &tau_f, from)
    };

    // λ = 2 Hz ..= 2 kHz
    let mut storage = DataStorage::new("./out/whitebox_rigid_body_lambda.csv").unwrap();
    for lambda_hz in [2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0, 1000.0, 2000.0] {
        let [jh, dh] = identify(lambda_hz);
        let (ej, ed) = error([jh, dh]);
        println!("λ = {lambda_hz:6.0} Hz:   J = {jh:.5} ({ej:+7.2} %), D = {dh:.5} ({ed:+7.2} %)");
        storage.add(&[lambda_hz, jh, dh, ej, ed]).unwrap();
    }
    storage.close().unwrap();

    // Validation of the rigid-body model of λ = 10 Hz (within the excited band, well below the
    // resonance; chosen without the true values). Below the resonance the model leaves out its
    // quasi-static tail, a gain error `(f / fr)^2` (0.1 % at 10 Hz, 1 % at 30 Hz) and a phase
    // error `2 ζ f / fr`: small, but far above the noise of a position measured this precisely
    // (as the error of `D`, 0.1 %, at the lowest lines), so that the coherence test rejects the
    // model in the band 0 ..= 30 Hz too. Whether such an error is acceptable is a tolerance, which
    // the test does not decide: the frequency response shows how large it is
    let chosen_hz = 10.0;
    let [jh, dh] = identify(chosen_hz);
    let model = TransferFunction::continuous(&[1.0], &[jh, dh, 0.0]);
    let validation = Validation::continuous(&model, TS, &tau, &theta).unwrap().evaluated_from(2.0).unwrap();
    let report = validation
        .check(
            &[
                Check::Coherence(CoherenceCheck::new(1.0, 1e-2)),
                Check::Coherence(CoherenceCheck::new(1.0, 1e-2).set_band((0.0, 30.0))),
            ],
            0.99,
        )
        .unwrap();
    println!("rigid-body model (λ = {chosen_hz} Hz) J = {jh:.5}, D = {dh:.5}, whole band and 0 ..= 30 Hz:\n{report}");
    let response = validation.frequency_response(1.0).unwrap().band((0.0, 30.0));
    println!("     f [Hz]   gain error [dB]   phase error [deg]  1 - (f / fr)^2 [dB]    2 ζ f / fr [deg]");
    for (i, f) in response.frequencies().iter().enumerate().filter(|(i, _)| i % 5 == 0 || *i == 1) {
        let (gain, phase) = (20.0 * (1.0 - (f / fr).powi(2)).log10(), (2.0 * zeta * f / fr).to_degrees());
        println!(
            "{f:11.0} {:17.4} {:19.4} {gain:18.4} {phase:19.4}",
            response.gain_error_db()[i],
            response.phase_error()[i].to_degrees()
        );
    }

    let mut storage = DataStorage::new("./out/whitebox_rigid_body.csv").unwrap();
    for k in 0..tau.len() {
        storage.add(&[k as f64 * TS, tau[k], theta[k], theta0[k]]).unwrap();
    }
    storage.close().unwrap();
}
