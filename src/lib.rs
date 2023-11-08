use ndarray::prelude::*;
use num::Float;

pub type StaticVector<T> = Array1<T>;
pub mod euler;

pub struct IVProblem<T> {
    pub t: T,
    pub y: StaticVector<T>,
    pub rhs: Box<dyn Fn(T, &StaticVector<T>) -> StaticVector<T>>,
}

impl<T: Float> IVProblem<T> {
    pub fn new(
        t: T,
        y: StaticVector<T>,
        rhs: impl Fn(T, &StaticVector<T>) -> StaticVector<T> + 'static,
    ) -> IVProblem<T> {
        IVProblem {
            t,
            y,
            rhs: Box::new(rhs),
        }
    }
}

pub trait IVPintegrator<T> {
    fn step_until(&self, problem: &mut IVProblem<T>, t_final: T);
}
