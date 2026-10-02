use thiserror::Error;

/// Errors of building or converting a `StateSpace`.
#[derive(Clone, Debug, Error, PartialEq)]
pub enum StateSpaceError {
    /// `A` is not square or is empty.
    #[error("System matrix A must be square and non-empty. Got {rows}×{cols}")]
    SystemMatrix {
        /// Rows of `A`.
        rows: usize,
        /// Columns of `A`.
        cols: usize
    },

    /// `B` does not have the expected size.
    #[error("Input matrix B must be {expected_row}×{expected_col}, but got {actual_rows}×{actual_cols}")]
    InputMatrix {
        /// Expected rows.
        expected_row: usize,
        /// Expected columns.
        expected_col: usize,
        /// Actual rows.
        actual_rows: usize,
        /// Actual columns.
        actual_cols: usize,
    },

    /// The input vector does not have one element per input.
    #[error("Input vector U must be {expected_row}, but got {actual_rows}")]
    InputVector {
        /// Expected length.
        expected_row: usize,
        /// Actual length.
        actual_rows: usize,
    },

    /// `C` does not have the expected size.
    #[error("Output matrixC must be {expected_row}×{expected_col}, but got {actual_rows}×{actual_cols}")]
    OutputMatrix {
        /// Expected rows.
        expected_row: usize,
        /// Expected columns.
        expected_col: usize,
        /// Actual rows.
        actual_rows: usize,
        /// Actual columns.
        actual_cols: usize,
    },

    /// `D` does not have the expected size.
    #[error("Feedthrough matrix D must be {expected_row}×{expected_col}, but got {actual_rows}×{actual_cols}")]
    FeedthroughMatrix {
        /// Expected rows.
        expected_row: usize,
        /// Expected columns.
        expected_col: usize,
        /// Actual rows.
        actual_rows: usize,
        /// Actual columns.
        actual_cols: usize,
    },

    /// The system has no states (e.g. a static gain).
    #[error("StateSpace must have at least one state")]
    EmptySystem,

    /// A matrix that must be inverted is singular.
    #[error("Matrix is singular")]
    SingularMatrix,

    /// The operation supports single-input single-output systems only.
    #[error("Only single-input single-output systems are supported, got {inputs} inputs and {outputs} outputs")]
    NotSiso {
        /// Number of inputs.
        inputs: usize,
        /// Number of outputs.
        outputs: usize,
    },

    /// The transfer function has more zeros than poles.
    #[error("Improper transfer function (numerator degree {numerator} > denominator degree {denominator}) has no state-space realization")]
    Improper {
        /// Numerator degree.
        numerator: usize,
        /// Denominator degree.
        denominator: usize,
    },
}
