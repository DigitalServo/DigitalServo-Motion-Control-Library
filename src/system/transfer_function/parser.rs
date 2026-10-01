//! Parse a transfer function from a rational expression in `s` (continuous) or `z` (discrete),
//! e.g. `"(s^2 + 10*s + 10) / (s^2 + 20*s + 100)"`. The variable must match the `Domain`
//! (`tf!` picks the `Domain` from the variable at compile time, see `__detect_domain`).
//!
//! Grammar (`^` binds tighter than unary minus, so `-s^2` is `-(s^2)`):
//! ```text
//! expr    := term (('+' | '-') term)*
//! term    := unary (('*' | '/') unary | primary)*   // juxtaposition = implicit '*', e.g. 10s, (s+1)(s+2)
//! unary   := ('+' | '-') unary | power
//! power   := primary ('^' ['+' | '-'] integer)?
//! primary := number | variable | '(' expr ')'
//! ```

use super::TransferFunction;
use crate::{Continuous, Discrete, Domain};
use crate::Polynomial;
use num_traits::Float;
use std::ops::AddAssign;
use std::str::FromStr;
use thiserror::Error;

/// Errors of parsing a transfer function from a string (`FromStr`, `tf!`).
#[derive(Clone, Debug, Error, PartialEq)]
pub enum TransferFunctionParseError {
    #[error("Unexpected character '{ch}' at position {pos}")]
    UnexpectedChar { ch: char, pos: usize },

    #[error("Invalid number '{text}' at position {pos}")]
    InvalidNumber { text: String, pos: usize },

    #[error("Unexpected token at position {pos}: expected {expected}")]
    UnexpectedToken { pos: usize, expected: &'static str },

    #[error("Unexpected end of input: expected {expected}")]
    UnexpectedEnd { expected: &'static str },

    #[error("Exponent must be an integer, got {value} at position {pos}")]
    NonIntegerExponent { value: f64, pos: usize },

    #[error("Division by zero")]
    DivisionByZero,

    #[error("Expression mixes 's' and 'z'")]
    MixedVariables,

    #[error("Variable '{found}' at position {pos} does not match the time domain (expected '{expected}')")]
    WrongVariable { expected: char, found: char, pos: usize },
}

type ParseResult<T> = Result<T, TransferFunctionParseError>;

#[derive(Clone, Copy, Debug, PartialEq)]
enum Token {
    Number(f64),
    Var,
    Plus,
    Minus,
    Star,
    Slash,
    Caret,
    LParen,
    RParen,
}

fn tokenize(src: &str, variable: char) -> ParseResult<Vec<(Token, usize)>> {
    let chars: Vec<char> = src.chars().collect();
    let mut tokens = Vec::new();
    // First occurrence of the other domain's variable (`z` when parsing `s`, and vice versa).
    let mut foreign: Option<(char, usize)> = None;
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        let token = match c {
            _ if c.is_whitespace() => {
                i += 1;
                continue;
            }
            '0'..='9' | '.' => {
                let start = i;
                while i < chars.len() && (chars[i].is_ascii_digit() || chars[i] == '.') {
                    i += 1;
                }
                // Exponent part (e.g. 1e-3), only if followed by digits.
                if i < chars.len() && (chars[i] == 'e' || chars[i] == 'E') {
                    let mut j = i + 1;
                    if j < chars.len() && (chars[j] == '+' || chars[j] == '-') {
                        j += 1;
                    }
                    if j < chars.len() && chars[j].is_ascii_digit() {
                        while j < chars.len() && chars[j].is_ascii_digit() {
                            j += 1;
                        }
                        i = j;
                    }
                }
                let text: String = chars[start..i].iter().collect();
                let value = text
                    .parse::<f64>()
                    .map_err(|_| TransferFunctionParseError::InvalidNumber { text, pos: start })?;
                tokens.push((Token::Number(value), start));
                continue;
            }
            _ if c == variable => Token::Var,
            's' | 'z' => {
                foreign = foreign.or(Some((c, i)));
                i += 1;
                continue;
            }
            '+' => Token::Plus,
            '-' => Token::Minus,
            '*' => Token::Star,
            '/' => Token::Slash,
            '^' => Token::Caret,
            '(' => Token::LParen,
            ')' => Token::RParen,
            _ => return Err(TransferFunctionParseError::UnexpectedChar { ch: c, pos: i }),
        };
        tokens.push((token, i));
        i += 1;
    }
    match foreign {
        None => Ok(tokens),
        Some(_) if tokens.iter().any(|&(t, _)| t == Token::Var) => {
            Err(TransferFunctionParseError::MixedVariables)
        }
        Some((found, pos)) => Err(TransferFunctionParseError::WrongVariable { expected: variable, found, pos }),
    }
}

/// Intermediate value: numerator / denominator, both descending-order polynomials in `s`.
#[derive(Clone, Debug)]
struct Rational<T> {
    num: Polynomial<T>,
    den: Polynomial<T>,
}

impl<T: Float + AddAssign> Rational<T> {
    fn constant(c: T) -> Self {
        Self { num: Polynomial(vec![c]), den: Polynomial(vec![T::one()]) }
    }

    fn var() -> Self {
        Self { num: Polynomial(vec![T::one(), T::zero()]), den: Polynomial(vec![T::one()]) }
    }

    fn neg(self) -> Self {
        Self { num: Polynomial(self.num.iter().map(|&c| -c).collect()), den: self.den }
    }

    fn add(self, rhs: Self) -> Self {
        // Skip the cross-multiplication in the common case of a shared denominator.
        if self.den == rhs.den {
            return Self { num: &self.num + &rhs.num, den: self.den };
        }
        let n1d2 = &self.num * &rhs.den;
        let n2d1 = &rhs.num * &self.den;
        Self { num: &n1d2 + &n2d1, den: &self.den * &rhs.den }
    }

    fn sub(self, rhs: Self) -> Self {
        self.add(rhs.neg())
    }

    fn mul(self, rhs: Self) -> Self {
        Self { num: &self.num * &rhs.num, den: &self.den * &rhs.den }
    }

    fn div(self, rhs: Self) -> ParseResult<Self> {
        if rhs.num.iter().all(|c| c.is_zero()) {
            return Err(TransferFunctionParseError::DivisionByZero);
        }
        Ok(Self { num: &self.num * &rhs.den, den: &self.den * &rhs.num })
    }

    fn pow(self, exp: i32) -> ParseResult<Self> {
        let base = if exp < 0 { Self::constant(T::one()).div(self)? } else { self };
        let mut result = Self::constant(T::one());
        for _ in 0..exp.unsigned_abs() {
            result = result.mul(base.clone());
        }
        Ok(result)
    }
}

struct Parser {
    tokens: Vec<(Token, usize)>,
    pos: usize,
}

impl Parser {
    fn peek(&self) -> Option<Token> {
        self.tokens.get(self.pos).map(|&(t, _)| t)
    }

    fn next(&mut self) -> Option<(Token, usize)> {
        let t = self.tokens.get(self.pos).copied();
        self.pos += 1;
        t
    }

    fn error(&self, expected: &'static str) -> TransferFunctionParseError {
        match self.tokens.get(self.pos) {
            Some(&(_, pos)) => TransferFunctionParseError::UnexpectedToken { pos, expected },
            None => TransferFunctionParseError::UnexpectedEnd { expected },
        }
    }

    fn expr<T: Float + AddAssign>(&mut self) -> ParseResult<Rational<T>> {
        let mut lhs = self.term()?;
        loop {
            match self.peek() {
                Some(Token::Plus) => {
                    self.next();
                    lhs = lhs.add(self.term()?);
                }
                Some(Token::Minus) => {
                    self.next();
                    lhs = lhs.sub(self.term()?);
                }
                _ => return Ok(lhs),
            }
        }
    }

    fn term<T: Float + AddAssign>(&mut self) -> ParseResult<Rational<T>> {
        let mut lhs = self.unary()?;
        loop {
            match self.peek() {
                Some(Token::Star) => {
                    self.next();
                    lhs = lhs.mul(self.unary()?);
                }
                Some(Token::Slash) => {
                    self.next();
                    lhs = lhs.div(self.unary()?)?;
                }
                // Implicit multiplication: `10s`, `2(s + 1)`, `(s + 1)(s + 2)`.
                Some(Token::Number(_) | Token::Var | Token::LParen) => {
                    lhs = lhs.mul(self.power()?);
                }
                _ => return Ok(lhs),
            }
        }
    }

    fn unary<T: Float + AddAssign>(&mut self) -> ParseResult<Rational<T>> {
        match self.peek() {
            Some(Token::Minus) => {
                self.next();
                Ok(self.unary()?.neg())
            }
            Some(Token::Plus) => {
                self.next();
                self.unary()
            }
            _ => self.power(),
        }
    }

    fn power<T: Float + AddAssign>(&mut self) -> ParseResult<Rational<T>> {
        let base = self.primary()?;
        if self.peek() != Some(Token::Caret) {
            return Ok(base);
        }
        self.next();
        let sign = match self.peek() {
            Some(Token::Minus) => {
                self.next();
                -1.0
            }
            Some(Token::Plus) => {
                self.next();
                1.0
            }
            _ => 1.0,
        };
        match self.next() {
            Some((Token::Number(v), pos)) => {
                let value = sign * v;
                if value.fract() != 0.0 || value.abs() > i32::MAX as f64 {
                    return Err(TransferFunctionParseError::NonIntegerExponent { value, pos });
                }
                base.pow(value as i32)
            }
            _ => {
                self.pos -= 1;
                Err(self.error("integer exponent"))
            }
        }
    }

    fn primary<T: Float + AddAssign>(&mut self) -> ParseResult<Rational<T>> {
        match self.next() {
            Some((Token::Number(v), pos)) => T::from(v)
                .map(Rational::constant)
                .ok_or(TransferFunctionParseError::InvalidNumber { text: v.to_string(), pos }),
            Some((Token::Var, _)) => Ok(Rational::var()),
            Some((Token::LParen, _)) => {
                let inner = self.expr()?;
                match self.next() {
                    Some((Token::RParen, _)) => Ok(inner),
                    _ => {
                        self.pos -= 1;
                        Err(self.error("')'"))
                    }
                }
            }
            _ => {
                self.pos -= 1;
                Err(self.error("number, variable or '('"))
            }
        }
    }
}

fn trim_leading_zeros<T: Float>(p: Polynomial<T>) -> Polynomial<T> {
    let coeffs: Vec<T> = p.0.into_iter().skip_while(|c| c.is_zero()).collect();
    if coeffs.is_empty() { Polynomial(vec![T::zero()]) } else { Polynomial(coeffs) }
}

impl<T: Float + AddAssign, D: Domain> TransferFunction<T, D> {
    /// Build a transfer function from a rational expression in `s` (`Continuous`) or `z`
    /// (`Discrete`). Common poles/zeros are cancelled (see `reduced`).
    /// Public entry points are `tf!` and `FromStr` (`"0.5 / (z - 0.5)".parse()`).
    fn parse(src: &str) -> Result<Self, TransferFunctionParseError> {
        let mut parser = Parser { tokens: tokenize(src, D::VARIABLE)?, pos: 0 };
        let rational: Rational<T> = parser.expr()?;
        if parser.pos < parser.tokens.len() {
            return Err(parser.error("end of input"));
        }
        Ok(Self::from_polynomials(trim_leading_zeros(rational.num), trim_leading_zeros(rational.den)).reduced())
    }

    /// Used by `tf!`: the float literal passed as `_hint` lets `T` fall back to `f64` when
    /// nothing else constrains it, while still allowing `f32` via a type annotation.
    #[doc(hidden)]
    pub fn __parse_with_hint(src: &str, _hint: T) -> Result<Self, TransferFunctionParseError> {
        Self::parse(src)
    }
}

/// Parse a string known only at runtime, with errors returned instead of panicking:
/// `"(s + 1) / (s + 2)".parse::<TransferFunction<f64>>()`.
impl<T: Float + AddAssign, D: Domain> FromStr for TransferFunction<T, D> {
    type Err = TransferFunctionParseError;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::parse(s)
    }
}

/// Compile-time domain detection for `tf!`: `0` if the expression uses only `s`, `1` if only `z`.
/// Panics (i.e. fails to compile when used in a const context) if it uses both or neither.
/// `{...}` placeholders of the `format!` string are skipped, so `{zeta}` does not count as `z`.
#[doc(hidden)]
pub const fn __detect_domain(src: &str) -> u8 {
    let bytes = src.as_bytes();
    let (mut has_s, mut has_z, mut in_placeholder) = (false, false, false);
    let mut i = 0;
    while i < bytes.len() {
        match bytes[i] {
            // `{{` / `}}` are escaped braces, not placeholders.
            b'{' if !in_placeholder && i + 1 < bytes.len() && bytes[i + 1] == b'{' => i += 1,
            b'}' if !in_placeholder && i + 1 < bytes.len() && bytes[i + 1] == b'}' => i += 1,
            b'{' => in_placeholder = true,
            b'}' => in_placeholder = false,
            b's' if !in_placeholder => has_s = true,
            b'z' if !in_placeholder => has_z = true,
            _ => {}
        }
        i += 1;
    }
    match (has_s, has_z) {
        (true, false) => 0,
        (false, true) => 1,
        (true, true) => panic!("tf!: expression mixes `s` and `z`"),
        (false, false) => panic!(
            "tf!: expression contains neither `s` nor `z`, so the time domain cannot be determined; \
             use `TransferFunction::continuous` / `TransferFunction::discrete` instead"
        ),
    }
}

/// Maps the result of `__detect_domain` to `Continuous` / `Discrete` at the type level.
#[doc(hidden)]
pub struct __DomainTag<const N: u8>;

#[doc(hidden)]
pub trait __SelectDomain {
    type Domain: Domain;
}

impl __SelectDomain for __DomainTag<0> {
    type Domain = Continuous;
}

impl __SelectDomain for __DomainTag<1> {
    type Domain = Discrete;
}

/// Write a transfer function as a string. The time domain is detected from the variable at
/// compile time: `s` gives `Continuous`, `z` gives `Discrete`.
/// ```ignore
/// let g = tf!("(s^2 + 10s + 10) / (s^2 + 20s + 100)"); // TransferFunction<f64, Continuous>
/// let h = tf!("0.5 / (z - 0.5)");                      // TransferFunction<f64, Discrete>
/// let k: TransferFunction<f32> = tf!("1 / (s + 1)");    // other float types via annotation
/// ```
/// Mixing `s` and `z`, or using neither (a constant), is a compile error.
///
/// The string is a `format!` string, so values can be embedded:
/// ```ignore
/// let g = 100.0;
/// let a = tf!("{g} / (s + {g})");
/// let b = tf!("{} / (s + {})", g, 2.0 * g);
/// ```
/// (`f64`/`f32` `Display` prints the shortest string that reads back to the same value, so
/// embedding does not lose precision.) Only the literal is inspected for `s`/`z`; the embedded
/// values are expected to be numbers.
///
/// The tokens can also be written directly:
/// ```ignore
/// let g = tf!((s^2 + 10.0 * s + 10.0) / (s^2 + 20.0 * s + 100.0));
/// ```
/// In that form the tokens are stringified before parsing, so `^` means power (not XOR).
/// Panics if the expression is invalid; use `str::parse` (`FromStr`) to handle errors.
#[macro_export]
macro_rules! tf {
    ($fmt:literal $(, $arg:expr)* $(,)?) => {
        $crate::TransferFunction::<
            _,
            <$crate::__DomainTag<{ $crate::__detect_domain($fmt) }> as $crate::__SelectDomain>::Domain,
        >::__parse_with_hint(&format!($fmt $(, $arg)*), 0.0)
            .unwrap_or_else(|e| panic!("tf!: {}", e))
    };
    ($($expr:tt)+) => {
        $crate::TransferFunction::<
            _,
            <$crate::__DomainTag<{ $crate::__detect_domain(stringify!($($expr)+)) }> as $crate::__SelectDomain>::Domain,
        >::__parse_with_hint(stringify!($($expr)+), 0.0)
            .unwrap_or_else(|e| panic!("tf!: {}", e))
    };
}
