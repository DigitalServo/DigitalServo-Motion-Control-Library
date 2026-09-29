//! Parse a transfer function from a rational expression in `s`, e.g.
//! `"(s^2 + 10*s + 10) / (s^2 + 20*s + 100)"`.
//!
//! Grammar (`^` binds tighter than unary minus, so `-s^2` is `-(s^2)`):
//! ```text
//! expr    := term (('+' | '-') term)*
//! term    := unary (('*' | '/') unary | primary)*   // juxtaposition = implicit '*', e.g. 10s, (s+1)(s+2)
//! unary   := ('+' | '-') unary | power
//! power   := primary ('^' ['+' | '-'] integer)?
//! primary := number | 's' | '(' expr ')'
//! ```

use super::TransferFunction;
use crate::Polynomial;
use num_traits::Float;
use std::ops::AddAssign;
use std::str::FromStr;
use thiserror::Error;

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

fn tokenize(src: &str) -> ParseResult<Vec<(Token, usize)>> {
    let chars: Vec<char> = src.chars().collect();
    let mut tokens = Vec::new();
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
            's' => Token::Var,
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
    Ok(tokens)
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
                Err(self.error("number, 's' or '('"))
            }
        }
    }
}

fn trim_leading_zeros<T: Float>(p: Polynomial<T>) -> Polynomial<T> {
    let coeffs: Vec<T> = p.0.into_iter().skip_while(|c| c.is_zero()).collect();
    if coeffs.is_empty() { Polynomial(vec![T::zero()]) } else { Polynomial(coeffs) }
}

impl<T: Float + AddAssign> TransferFunction<T> {
    /// Build a transfer function from a rational expression in `s`, e.g.
    /// `TransferFunction::<f64>::parse("(s^2 + 10*s + 10) / (s^2 + 20*s + 100)")`.
    /// Common poles/zeros are cancelled (see `reduced`).
    pub fn parse(src: &str) -> Result<Self, TransferFunctionParseError> {
        let mut parser = Parser { tokens: tokenize(src)?, pos: 0 };
        let rational: Rational<T> = parser.expr()?;
        if parser.pos < parser.tokens.len() {
            return Err(parser.error("end of input"));
        }
        Ok(Self {
            numerator: trim_leading_zeros(rational.num),
            denominator: trim_leading_zeros(rational.den),
        }
        .reduced())
    }
}

impl<T: Float + AddAssign> FromStr for TransferFunction<T> {
    type Err = TransferFunctionParseError;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::parse(s)
    }
}

/// Write a transfer function directly as a rational expression in `s`:
/// ```ignore
/// let g: TransferFunction<f64> = tf!((s^2 + 10.0 * s + 10.0) / (s^2 + 20.0 * s + 100.0));
/// ```
/// The tokens are stringified and handed to `TransferFunction::parse`, so `^` means power here
/// (not XOR). Panics if the expression is invalid; use `TransferFunction::parse` to handle errors.
#[macro_export]
macro_rules! tf {
    ($($expr:tt)+) => {
        $crate::TransferFunction::parse(stringify!($($expr)+))
            .unwrap_or_else(|e| panic!("tf!: {}", e))
    };
}
