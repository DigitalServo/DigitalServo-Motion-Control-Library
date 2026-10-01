pub mod mt;
pub mod ms;
pub mod mcv;
pub mod sin;
pub mod polynomial;
pub mod cycloid;

pub struct TrajectoryProfile<T> {
    pub s: T,
    pub v: T,
    pub a: T
}
