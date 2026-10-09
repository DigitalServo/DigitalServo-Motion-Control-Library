//! Integration tests in one binary: one link, and one malware scan by macOS on the first run of
//! a newly linked binary (a few seconds each), instead of one per file. Run a single module with
//! `cargo test --test integration <module>::`.

mod discretize;
mod excitation;
mod frequency_transfer_function;
mod logger;
mod math;
mod partial_fraction;
mod ptc_output;
mod root_locus;
mod signal;
mod spectrum;
mod state_space;
mod stable_inverse;
mod status;
mod system_identification;
mod time_expression;
mod trajectory;
mod transfer_function;
mod validation;
mod welch;
