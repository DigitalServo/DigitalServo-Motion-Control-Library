use thiserror::Error;

#[derive(Clone, Debug, Error, PartialEq)]
pub enum StateSpaceError {
    #[error("System matrix A must be square and non-empty. Got {rows}×{cols}")]
    SystemMatrix {
        rows: usize,
        cols: usize
    },

    #[error("Input matrix B must be {expected_row}×{expected_col}, but got {actual_rows}×{actual_cols}")]
    InputMatrix {
        expected_row: usize,
        expected_col: usize,
        actual_rows: usize,
        actual_cols: usize,
    },

    #[error("Input vector U must be {expected_row}, but got {actual_rows}")]
    InputVector {
        expected_row: usize,
        actual_rows: usize,
    },

    #[error("Output matrixC must be {expected_row}×{expected_col}, but got {actual_rows}×{actual_cols}")]
    OutputMatrix {
        expected_row: usize,
        expected_col: usize,
        actual_rows: usize,
        actual_cols: usize,
    },

    #[error("Feedthrough matrix D must be {expected_row}×{expected_col}, but got {actual_rows}×{actual_cols}")]
    FeedthroughMatrix {
        expected_row: usize,
        expected_col: usize,
        actual_rows: usize,
        actual_cols: usize,
    },

    #[error("StateSpace must have at least one state")]
    EmptySystem,

    #[error("Matrix is singular")]
    SingularMatrix,
}
