//! Frequency-domain SML identification of a two-inertia plant with a dead time from a noisy
//! periodic record, as an application would do it:
//!
//! 1. the period detected from the input (`SmlIdentifier::new`, as the next two steps),
//! 2. the excited lines and the SML data of the steady-state periods,
//! 3. the band selected and the zero-order hold compensated,
//! 4. the structure (orders `n`, `m`, integrators `q`) and the dead time chosen by the BIC
//!    (`SmlIdentifier::search`: the fewest parameters within 2 of the lowest BIC), with the cost
//!    test shown for comparison (it passes several structures),
//! 5. the measured response with its standard deviation, the model and the cost of each line
//!    written to `out/sml.csv`.
//!
//! `cargo run --release --example sml`

use std::f64::consts::PI;

use dsmc::logger::DataStorage;
use dsmc::signal::excitation::multisine;
use dsmc::system_identification::frequency_response::sml::{SmlIdentifier, SmlIdentifierOptions, SmlSearchOptions, SmlStructure};
use dsmc::{TransferFunction, TransferFunctionWithDelay};
use serde::Serialize;

#[derive(Serialize)]
struct Row {
    freq_hz: f64,
    measured_gain_db: f64,
    measured_phase_deg: f64,
    /// Standard deviation of the measured response relative to its magnitude.
    relative_std: f64,
    model_gain_db: f64,
    model_phase_deg: f64,
    /// `|e_k|^2`, expected about `expected_cost / F`.
    line_cost: f64,
}

/// Standard normal pseudo-random numbers (xorshift64, Box-Muller).
fn gaussian_source(mut state: u64) -> impl FnMut() -> f64 {
    move || {
        let mut uniform = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            ((state >> 11) as f64 + 0.5) / (1u64 << 53) as f64
        };
        (-2.0 * uniform().ln()).sqrt() * (2.0 * PI * uniform()).cos()
    }
}

fn main() {
    // Plant: rigid body 1 / (J s^2) with an antiresonance at 30 Hz, a resonance at 50 Hz and a
    // dead time of 2 ms, behind a zero-order hold at 2 kHz
    let ts = 5e-4;
    let (wz, wp, zeta, inertia) = (2.0 * PI * 30.0, 2.0 * PI * 50.0, 0.05, 1e-3);
    let gain = wp * wp / (wz * wz) / inertia;
    let plant = TransferFunctionWithDelay::new(
        TransferFunction::continuous(&[gain, gain * 2.0 * zeta * wz, gain * wz * wz], &[1.0, 2.0 * zeta * wp, wp * wp, 0.0, 0.0]),
        0.002,
    );

    // Multisine on lines 1 ..= 205 (up to 100 Hz) of 4096 samples, flat at the output; 2 transient
    // periods, 8 periods and an incomplete one; noise of 0.01 % / 2 % of the RMS on the input /
    // output
    let period = 4096;
    let f0 = 1.0 / (period as f64 * ts);
    let harmonics: Vec<usize> = (1..=205).collect();
    let total = 10 * period + 1000;
    let u0: Vec<f64> = multisine(total as f64 * ts, ts, f0, &harmonics, |k| 1.0 / plant.frequency_response(2.0 * PI * f0 * k as f64).norm(), 1).unwrap();
    let y0 = plant.simulate(ts, &u0).unwrap();
    let rms = |x: &[f64]| (x.iter().map(|v| v * v).sum::<f64>() / x.len() as f64).sqrt();
    let (su, sy) = (1e-4 * rms(&u0), 0.02 * rms(&y0[2 * period..]));
    let mut gaussian = gaussian_source(0x2545_f491_4f6c_dd1d);
    let u: Vec<f64> = u0.iter().map(|v| v + su * gaussian()).collect();
    let y: Vec<f64> = y0.iter().map(|v| v + sy * gaussian()).collect();

    // 1.-3. Period, excited lines, steady-state periods, band 0.4 .. 100 Hz, hold compensated
    let record = SmlIdentifierOptions { transient_periods: 2, band: Some((0.4, 100.0)), ..Default::default() };
    let identifier = SmlIdentifier::new(&u, &y, ts, &record).unwrap();
    let lines = identifier.excited_lines();
    println!("period: {} samples ({:.3} s)", identifier.period(), identifier.period() as f64 * ts);
    println!("{} excited lines ({} .. {}), {} in the band, {} periods", lines.len(), lines[0], lines[lines.len() - 1], identifier.data().lines.len(), identifier.data().periods);

    // 4. Structure and dead time by the BIC: every (n, m, q) with n <= 4, m <= 2, q <= 2 and dead
    // times 0 .. 4 ms; of the structures within 2 of the lowest BIC, the fewest parameters
    let structures = SmlStructure::grid(0..=4, 0..=2, 0..=2, (0..=8).map(|k| k as f64 * ts));
    let search = identifier.search(&structures, &SmlSearchOptions::default());
    println!("\n{} candidates; the 10 lowest BIC:", search.candidates.len());
    println!("  n  m  q  delay[ms]        cost    expected ±    std   test        BIC");
    for c in search.ranked().iter().take(10) {
        let (s, r) = (&c.structure, &c.fit().unwrap().result);
        let test = if r.cost_test(2.0) { "pass" } else { "fail" };
        println!("{:3} {:2} {:2} {:10.1} {:11.1} {:11.1} ± {:6.1}   {test} {:10.1}", s.denominator_order, s.numerator_order, s.integrators, s.delay * 1e3, r.cost, r.expected_cost, r.cost_std, r.bic());
    }
    let selected = search.selected().expect("no structure identified");
    let best = selected.fit().unwrap();
    let r = &best.result;
    println!("\nselected {} (nθ = {}), BIC {:.1}, {} iterations ({:?})", selected.structure, r.parameters, r.bic(), r.iterations, r.termination);
    println!("  numerator   {:?}\n          ± {:?}", best.model.tf.numerator.0, r.std_numerator);
    println!("  denominator {:?}\n          ± {:?}", best.model.tf.denominator.0, r.std_denominator);
    println!("  poles {:?}", best.model.tf.pz_map().poles);
    println!("true\n  numerator   {:?}\n  denominator {:?}\n  delay {:.1} ms", plant.tf.numerator.0, plant.tf.denominator.0, plant.delay * 1e3);

    // 5. Lines, model and line costs
    let costs = identifier.line_costs(best);
    let db = |g: f64| 20.0 * g.log10();
    let mut storage = DataStorage::new("./out/sml.csv").unwrap().set_header([
        "freq_hz",
        "measured_gain_db",
        "measured_phase_deg",
        "relative_std",
        "model_gain_db",
        "model_phase_deg",
        "line_cost",
    ]);
    for (line, &line_cost) in identifier.continuous_data().lines.iter().zip(&costs) {
        let (measured, model) = (line.response(), best.model.frequency_response(line.omega));
        storage
            .add(&Row {
                freq_hz: line.omega / (2.0 * PI),
                measured_gain_db: db(measured.norm()),
                measured_phase_deg: measured.arg().to_degrees(),
                relative_std: line.response_std() / measured.norm(),
                model_gain_db: db(model.norm()),
                model_phase_deg: model.arg().to_degrees(),
                line_cost,
            })
            .unwrap();
    }
    storage.close().unwrap();
    println!("\nout/sml.csv: {} lines (line costs: mean {:.2}, expected about {:.2})", costs.len(), costs.iter().sum::<f64>() / costs.len() as f64, r.expected_cost / r.lines as f64);
}
