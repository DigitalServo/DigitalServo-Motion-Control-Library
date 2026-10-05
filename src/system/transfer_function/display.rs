//! `Display` implementations for `TransferFunction` and `PzMap`.

use super::{PzMap, TransferFunction};
use crate::Domain;
use crate::Polynomial;
use num_complex::Complex;
use num_traits::Float;

impl<T: Float + std::fmt::Display, D> std::fmt::Display for PzMap<T, D> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Honor the precision given by the caller (e.g. `{:.2}`).
        let write_complex = |f: &mut std::fmt::Formatter<'_>, c: &Complex<T>| {
            let sign = if c.im.is_sign_negative() { '-' } else { '+' };
            match f.precision() {
                Some(p) => writeln!(f, "  {:.*} {} j{:.*}", p, c.re, sign, p, c.im.abs()),
                None => writeln!(f, "  {} {} j{}", c.re, sign, c.im.abs()),
            }
        };
        writeln!(f, "Poles:")?;
        for pole in &self.poles {
            write_complex(f, pole)?;
        }
        writeln!(f, "Zeros:")?;
        for zero in &self.zeros {
            write_complex(f, zero)?;
        }
        Ok(())
    }
}

impl<T: Float + std::fmt::Display, D: Domain> std::fmt::Display for TransferFunction<T, D> {
    /// e.g. `1.0 / (s + 2.0)` or `0.5 / (z - 0.5)`. The output can be read back with `str::parse` (`FromStr`).
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let numer = format_polynomial(&self.numerator, D::VARIABLE, f.precision());
        let denom = format_polynomial(&self.denominator, D::VARIABLE, f.precision());
        let paren = |s: String, terms: usize| if terms > 1 { format!("({})", s) } else { s };

        let denom_terms = count_terms(&self.denominator);
        let denom_is_one = denom_terms <= 1 && self.denominator.iter().next_back() == Some(&T::one());
        let numer = paren(numer, if denom_is_one { 0 } else { count_terms(&self.numerator) });
        if denom_is_one {
            write!(f, "{}", numer)
        } else {
            write!(f, "{} / {}", numer, paren(denom, denom_terms))
        }
    }
}

fn count_terms<T: Float>(p: &Polynomial<T>) -> usize {
    p.iter().filter(|c| !c.is_zero()).count()
}

/// Descending polynomial in `variable`, e.g. `s^2 - 3.0 * s + 2.0`. Without an explicit precision,
/// coefficients get a trailing `.0` when integral, so that `1.0` is printed as `1.0` rather than `1`.
fn format_polynomial<T: Float + std::fmt::Display>(
    p: &Polynomial<T>,
    variable: char,
    precision: Option<usize>,
) -> String {
    let fmt_num = |c: T| match precision {
        Some(prec) => format!("{:.*}", prec, c),
        None => {
            // Float `Display` never uses exponent notation, so a finite value without '.' is integral.
            let s = c.to_string();
            if c.is_finite() && !s.contains('.') { s + ".0" } else { s }
        }
    };

    let degree = p.len().saturating_sub(1);
    let mut out = String::new();
    for (i, &c) in p.iter().enumerate() {
        if c.is_zero() {
            continue;
        }
        let power = degree - i;
        let sign = if c.is_sign_negative() { "-" } else { "+" };
        let abs = c.abs();

        let var = match power {
            0 => String::new(),
            1 => variable.to_string(),
            n => format!("{}^{}", variable, n),
        };
        let term = if power == 0 {
            fmt_num(abs)
        } else if abs == T::one() {
            var
        } else {
            format!("{} * {}", fmt_num(abs), var)
        };

        if out.is_empty() {
            if sign == "-" {
                out.push('-');
            }
        } else {
            out.push_str(&format!(" {} ", sign));
        }
        out.push_str(&term);
    }

    if out.is_empty() { fmt_num(T::zero()) } else { out }
}
