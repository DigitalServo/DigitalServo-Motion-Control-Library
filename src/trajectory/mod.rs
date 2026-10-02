//! Rest-to-rest motion profiles (trajectories).

pub mod mt;
pub mod ms;
pub mod mcv;
pub mod sin;
pub mod smoothstep;
pub mod cycloid;

pub use cycloid::Cycloid;
pub use mcv::ModifiedConstantVelocity;
pub use ms::ModifiedSine;
pub use mt::ModifiedTrapezoid;
use serde::Serialize;
pub use smoothstep::SmoothPolynomial;
pub use sin::Sin;

use num_traits::{Float, FloatConst};

/// Position, velocity and acceleration of a trajectory at one instant (see `Trajectory::profile`).
#[derive(Debug, Clone, Serialize)]
pub struct TrajectoryProfile<T> {
    /// Position.
    pub s: T,
    /// Velocity.
    pub v: T,
    /// Acceleration.
    pub a: T
}

impl<T: Float> TrajectoryProfile<T> {
    /// At rest at position `s`.
    pub fn rest(s: T) -> Self {
        Self { s, v: T::zero(), a: T::zero() }
    }
}

/// Rest-to-rest motion profile from 0 to `distance` over the normalized time `x = 0..1`.
/// Parameters specific to a profile (e.g. the constant-velocity share of `ModifiedConstantVelocity`)
/// are fields of the implementing type, so every profile is used the same way.
///
/// ```
/// use dsmc::trajectory::{ModifiedSine, Trajectory};
///
/// let duration = 0.1;
/// let samples = ModifiedSine.generate(1.0_f64, 101);
/// // Physical velocity at the middle of the move
/// let v = samples[50].v / duration;
/// ```
pub trait Trajectory<T: Float> {
    /// Position, velocity and acceleration at normalized time `x` (derivatives with respect to `x`;
    /// divide by `duration` / `duration^2` for physical units). At rest for `x < 0` and `x > 1`;
    /// `x = 1` is evaluated on the move (left-hand value where the acceleration jumps).
    fn profile(&self, distance: T, x: T) -> TrajectoryProfile<T>;

    /// `profile` at `x = i / (samples - 1)` for `i = 0..samples`.
    fn generate(&self, distance: T, samples: usize) -> Vec<TrajectoryProfile<T>> {
        let last = T::from(samples.saturating_sub(1).max(1)).unwrap();
        (0..samples)
            .map(|i| self.profile(distance, T::from(i).unwrap() / last))
            .collect()
    }
}

/// Any of the profiles of this module, selected by value (e.g. at runtime or in a list) without
/// boxing. For bounds on what a profile can do (e.g. `ReferenceTrajectory` for the reference of
/// perfect tracking control), use the individual types instead.
///
/// ```ignore
/// let profiles = [
///     TrajectoryKind::Sin,
///     TrajectoryKind::ModifiedConstantVelocity { constant_velocity_percent: 30.0 },
///     TrajectoryKind::SmoothPolynomial { smoothness: 4 },
/// ];
/// for p in &profiles {
///     let samples = p.generate(1.0, 500);
/// }
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub enum TrajectoryKind {
    /// [`Sin`].
    Sin,
    /// [`Cycloid`].
    Cycloid,
    /// [`ModifiedTrapezoid`].
    ModifiedTrapezoid,
    /// [`ModifiedSine`].
    ModifiedSine,
    /// [`ModifiedConstantVelocity`].
    ModifiedConstantVelocity {
        /// Share of the move at constant velocity \[%\].
        constant_velocity_percent: f64,
    },
    /// [`SmoothPolynomial`].
    SmoothPolynomial {
        /// Number of derivatives that vanish at both ends (`k`).
        smoothness: usize,
    },
}

impl<T: Float + FloatConst> Trajectory<T> for TrajectoryKind {
    fn profile(&self, distance: T, x: T) -> TrajectoryProfile<T> {
        match *self {
            Self::Sin => Sin.profile(distance, x),
            Self::Cycloid => Cycloid.profile(distance, x),
            Self::ModifiedTrapezoid => ModifiedTrapezoid.profile(distance, x),
            Self::ModifiedSine => ModifiedSine.profile(distance, x),
            Self::ModifiedConstantVelocity { constant_velocity_percent } => {
                ModifiedConstantVelocity { constant_velocity_percent }.profile(distance, x)
            }
            Self::SmoothPolynomial { smoothness } => SmoothPolynomial { smoothness }.profile(distance, x),
        }
    }
}

impl From<Sin> for TrajectoryKind {
    fn from(_: Sin) -> Self {
        Self::Sin
    }
}

impl From<Cycloid> for TrajectoryKind {
    fn from(_: Cycloid) -> Self {
        Self::Cycloid
    }
}

impl From<ModifiedTrapezoid> for TrajectoryKind {
    fn from(_: ModifiedTrapezoid) -> Self {
        Self::ModifiedTrapezoid
    }
}

impl From<ModifiedSine> for TrajectoryKind {
    fn from(_: ModifiedSine) -> Self {
        Self::ModifiedSine
    }
}

impl From<ModifiedConstantVelocity> for TrajectoryKind {
    fn from(p: ModifiedConstantVelocity) -> Self {
        Self::ModifiedConstantVelocity { constant_velocity_percent: p.constant_velocity_percent }
    }
}

impl From<SmoothPolynomial> for TrajectoryKind {
    fn from(p: SmoothPolynomial) -> Self {
        Self::SmoothPolynomial { smoothness: p.smoothness }
    }
}

/// Profiles usable as a reference: they have an exact continuous-time representation on the time
/// axis (`DelayedRationalSum`, `PiecewisePolynomial`), as needed e.g. for the output reference of
/// perfect tracking control by stable inversion (`Reference: feedforward::ptc::ReferenceSignal`).
pub trait ReferenceTrajectory<T: Float> {
    /// Exact continuous-time representation of the reference.
    type Reference;

    /// A move of `distance` in `duration` \[s\] starting at `start` \[s\] (0 before, `distance` after).
    fn reference(&self, distance: T, duration: T, start: T) -> Self::Reference;
}
