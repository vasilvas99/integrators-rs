use ndarray::prelude::*;
use num::Float;

type Vector<T> = Array1<T>;
type RhsFunc<T> = fn(T, &Vector<T>) -> Vector<T>;

pub mod euler;

pub struct IVProblem<T> {
    pub t: T,
    pub y: Vector<T>,
    pub rhs: RhsFunc<T>,
}

impl<T: Float> IVProblem<T> {
    pub fn new(t: T, y: Vector<T>, rhs: RhsFunc<T>) -> IVProblem<T> {
        IVProblem { t, y, rhs }
    }
}

pub trait IVPintegrator<T> {
    fn step_until(&self, problem: &mut IVProblem<T>, t_final: T);
}
