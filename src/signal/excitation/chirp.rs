//! Swept sines.

use num_traits::{Float, FloatConst};

use super::{duration_error, ExcitationError};
use crate::FrequencyTransferFunction;
use crate::sampling::whole_samples;

/// Linear chirp (swept sine) of unit amplitude from `f0` to `f1` \[Hz\] over `tlen` \[s\] sampled
/// with period `ts` \[s\] (`tlen` a whole number of samples):
///
/// ```text
/// u[k] = sin(2π (f0 t + (f1 - f0) t^2 / (2 tlen))),   t = k ts
/// ```
///
/// Not periodic: validate with `Validation::coherence_test` / `cross_correlation`, not `line_test`.
pub fn chirp<T: Float>(
    tlen: T,
    ts: T,
    f0: T,
    f1: T
) -> Result<Vec<T>, ExcitationError> {
    let n_samples = whole_samples(tlen, ts).map_err(|kind| duration_error(kind, tlen, ts))?;
    let (ts, f0, f1) = (ts.to_f64().unwrap(), f0.to_f64().unwrap(), f1.to_f64().unwrap());
    let duration = n_samples as f64 * ts;
    Ok((0..n_samples)
        .map(|k| {
            let t = k as f64 * ts;
            T::from((2.0 * std::f64::consts::PI * (f0 * t + (f1 - f0) * t * t / (2.0 * duration))).sin()).unwrap()
        })
        .collect())
}

/// `chirp` with the amplitude `|F(j ω(t))|` of the filter `filter` at the instantaneous frequency
/// `ω(t) = 2π (f0 + (f1 - f0) t / tlen)`, scaled to unit RMS:
///
/// ```text
/// u[k] = c |F(j ω(t))| sin(2π (f0 t + (f1 - f0) t^2 / (2 tlen))),   t = k ts
/// c    = 1 / sqrt(Σ_k (|F(j ω(t))| sin(…))^2 / n)
/// ```
///
/// Quasi-static: the spectrum follows `|F|` only if the sweep is slow, i.e. the envelope changes
/// little within the time constants of `F`. Across a resonance of half-power bandwidth `Δf` [Hz]
/// the sweep rate must be well below `Δf^2`: `|f1 - f0| / tlen << Δf^2` (the line takes `Δf / rate`
/// to cross the resonance, which must be long against its decay time `1 / (π Δf)`). A faster sweep
/// smears the shape (the modulated envelope spreads the power to the neighbouring frequencies),
/// and sharp peaks and notches of `|F|` are flattened most.
///
/// Not periodic: validate with `Validation::coherence_test` / `cross_correlation`, not `line_test`.
/// `ExcitationError::NoPower` if there are no samples, or `|F|` is zero over the sweep or not finite.
pub fn shaped_chirp<T: Float + FloatConst + 'static>(
    filter: impl Into<FrequencyTransferFunction<T>>,
    tlen: T,
    ts: T,
    f0: T,
    f1: T,
) -> Result<Vec<T>, ExcitationError> {
    let filter = filter.into();
    let n_samples = whole_samples(tlen, ts).map_err(|kind| duration_error(kind, tlen, ts))?;
    let (ts, f0, f1) = (ts.to_f64().unwrap(), f0.to_f64().unwrap(), f1.to_f64().unwrap());
    let duration = n_samples as f64 * ts;
    let u: Vec<f64> = (0..n_samples)
        .map(|k| {
            let t = k as f64 * ts;
            let phase = 2.0 * std::f64::consts::PI * (f0 * t + (f1 - f0) * t * t / (2.0 * duration));
            let omega = 2.0 * std::f64::consts::PI * (f0 + (f1 - f0) * t / duration);
            let gain = filter.response(T::from(omega).unwrap()).norm().to_f64().unwrap_or(f64::NAN);
            gain * phase.sin()
        })
        .collect();
    let power = u.iter().map(|v| v * v).sum::<f64>() / n_samples as f64;
    if !(power > 0.0 && power.is_finite()) {
        return Err(ExcitationError::NoPower);
    }
    let scale = 1.0 / power.sqrt();
    Ok(u.iter().map(|v| T::from(scale * v).unwrap()).collect())
}
