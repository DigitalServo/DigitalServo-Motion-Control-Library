mod error;
pub use error::StateSpaceError;

use nalgebra::DMatrix;

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct StateSpaceOrder {
    pub system: usize,
    pub input: usize,
    pub output: usize,
}

#[derive(Clone, Debug)]
pub struct StateSpace<T> {
    pub a: DMatrix<T>,
    pub b: DMatrix<T>,
    pub c: DMatrix<T>,
    pub d: DMatrix<T>,
    pub order: StateSpaceOrder,
}

impl<T> StateSpace<T> {
    pub fn new(a: DMatrix<T>, b: DMatrix<T>, c: DMatrix<T>, d: DMatrix<T>) -> Result<Self, StateSpaceError> {

        let state_order = a.nrows();
        let input_order = b.ncols();
        let output_order = c.nrows();

        if a.ncols() != state_order {
            return Err(StateSpaceError::SystemMatrix {
                rows: a.nrows(),
                cols: a.ncols(),
            })
        }

        if b.nrows() != state_order {
            return Err(StateSpaceError::InputMatrix {
                expected_row: state_order,
                expected_col: input_order,
                actual_rows: b.nrows(),
                actual_cols: b.ncols(),
            })
        }

        if c.ncols() != state_order {
            return Err(StateSpaceError::OutputMatrix {
                expected_row: output_order,
                expected_col: state_order,
                actual_rows: c.nrows(),
                actual_cols: c.ncols(),
            })
        }

        if (d.nrows(), d.ncols()) != (output_order, input_order) {
            return Err(StateSpaceError::FeedthroughMatrix {
                expected_row: output_order,
                expected_col: input_order,
                actual_rows: d.nrows(),
                actual_cols: d.ncols(),
            })
        }

        let order = StateSpaceOrder {
            system: a.nrows(),
            input: b.ncols(),
            output: c.nrows(),
        };

        Ok(Self {a, b, c, d, order})
    }
}
