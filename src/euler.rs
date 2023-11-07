use num::Float;

use crate::IVPintegrator;
use crate::IVProblem;
use ndarray::ScalarOperand;

pub struct ExplicitEuler<T> {
    pub(crate) step: T,
}

impl<T: Float + ScalarOperand> ExplicitEuler<T> {
    pub fn new(step: T) -> ExplicitEuler<T> {
        ExplicitEuler { step }
    }
}

impl<T: Float + ScalarOperand> IVPintegrator<T> for ExplicitEuler<T> {
    fn step_until(&self, problem: &mut IVProblem<T>, t_final: T) {
        if t_final <= problem.t {
            return;
        }

        // Safety: ceil ensures that number can be represented as int
        let num_steps = ((t_final - problem.t) / self.step).ceil().to_i64().unwrap();

        for _ in 0..num_steps {
            problem.y = &problem.y + (problem.rhs)(problem.t, &problem.y) * self.step;
            problem.t = problem.t + self.step;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Vector;
    use ndarray::Array;

    const T_MAX_LOGISTIC: f64 = 20.0;
    const STEP_LOGISTIC: f64 = 0.045;
    const Y0_LOGISTIC: f64 = 0.01;
    const K_CONST_LOGISTIC: f64 = 10.0;

    fn logistic_rhs(_t: f64, y: &Vector<f64>) -> Vector<f64> {
        let y_diff = -y + 1.0;
        return y * y_diff * K_CONST_LOGISTIC;
    }

    fn logistic_exact(t: f64) -> f64 {
        let denom = Y0_LOGISTIC + (1.0 - Y0_LOGISTIC) * f64::exp(-K_CONST_LOGISTIC * t);
        return Y0_LOGISTIC / denom;
    }

    #[test]
    fn explicit_euler_logistic() {
        let mut problem = IVProblem::new(0.0, Array::from_vec(vec![Y0_LOGISTIC]), logistic_rhs);
        let integrator = ExplicitEuler::new(STEP_LOGISTIC);
        integrator.step_until(&mut problem, T_MAX_LOGISTIC);

        let y_result = problem.y[0];
        let y_exact = logistic_exact(problem.t);
        let abs_error = (y_exact - y_result).abs();
        assert!(abs_error <= STEP_LOGISTIC * 10.0); // Euler method is first order, so error should be less than 10*step
        assert!((problem.t - T_MAX_LOGISTIC).abs() <= STEP_LOGISTIC);
    }
}
