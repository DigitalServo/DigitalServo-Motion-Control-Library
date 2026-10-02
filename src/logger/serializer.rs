//! Field serializers that write numbers with a fixed format (`#[serde(serialize_with = ...)]`).

use std::fmt::{Display, LowerExp};
use serde::Serializer;

/// Fixed-point with `PRECISION` decimals, e.g. `#[serde(serialize_with = "serialize_float::<3, _, _>")]`.
///
/// ```
/// use serde::Serialize;
/// use dsmc::logger::serializer::{serialize_float, serialize_float_exp, serialize_int};
///
/// #[derive(Serialize)]
/// struct Row {
///     #[serde(serialize_with = "serialize_int::<6, _, _>")]
///     step: usize,       // 000042
///     #[serde(serialize_with = "serialize_float::<3, _, _>")]
///     position: f64,     // 1.235
///     #[serde(serialize_with = "serialize_float_exp::<2, _, _>")]
///     error: f64,        // 1.23e-6
/// }
/// ```
pub fn serialize_float<const PRECISION: usize, T: Display, S: Serializer>(value: &T, serializer: S) -> Result<S::Ok, S::Error> {
    serializer.serialize_str(&format!("{:.1$}", value, PRECISION))
}

/// Exponential notation with `PRECISION` decimals in the mantissa.
pub fn serialize_float_exp<const PRECISION: usize, T: Display + LowerExp, S: Serializer>(value: &T, serializer: S) -> Result<S::Ok, S::Error> {
    serializer.serialize_str(&format!("{:.1$e}", value, PRECISION))
}

/// Zero-padded to at least `DIGITS` characters.
pub fn serialize_int<const DIGITS: usize, T: Display, S: Serializer>(value: &T, serializer: S) -> Result<S::Ok, S::Error> {
    serializer.serialize_str(&format!("{:01$}", value, DIGITS))
}
