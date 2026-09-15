use std::fmt::{Display, LowerExp};
use serde::Serializer;

pub fn serialize_float<const PRECISION: usize, T: Display, S: Serializer>(value: &T, serializer: S) -> Result<S::Ok, S::Error> {
    serializer.serialize_str(&format!("{:.1$}", value, PRECISION))
}

pub fn serialize_float_exp<const PRECISION: usize, T: Display + LowerExp, S: Serializer>(value: &T, serializer: S) -> Result<S::Ok, S::Error> {
    serializer.serialize_str(&format!("{:.1$e}", value, PRECISION))
}

pub fn serialize_int<const DIGITS: usize, T: Display, S: Serializer>(value: &T, serializer: S) -> Result<S::Ok, S::Error> {
    serializer.serialize_str(&format!("{:01$}", value, DIGITS))
}
