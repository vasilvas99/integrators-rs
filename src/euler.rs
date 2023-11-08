use num::Float;

use crate::IVPIntegrator;
use crate::IVProblem;
use ndarray::ScalarOperand;

pub struct ExplicitEuler<T> {
    pub(crate) step: T,
}

/// Implicit Euler Method implemented as predictor-corrector method PC11
pub struct PredictorCorrector11<T> {
    pub(crate) step: T,
    pub(crate) num_iters: u64,
}

/// Improved Euler (Crank–Nicolson) method implemented as predictor-corrector method PC12
pub struct PredictorCorrector12<T> {
    pub(crate) step: T,
    pub(crate) num_iters: u64,
}

impl<T: Float + ScalarOperand> ExplicitEuler<T> {
    pub fn new(step: T) -> ExplicitEuler<T> {
        ExplicitEuler { step }
    }
}

impl<T: Float + ScalarOperand> PredictorCorrector11<T> {
    /// Define initial step and number of predictor-corrector steps
    /// if num_iters is None, the default value that will be used is 3
    pub fn new(step: T, num_iters: Option<u64>) -> PredictorCorrector11<T> {
        let num_iters = num_iters.unwrap_or(3);
        PredictorCorrector11 { step, num_iters }
    }
}

impl<T: Float + ScalarOperand> PredictorCorrector12<T> {
    /// Define initial step and number of predictor-corrector steps
    /// if num_iters is None, the default value that will be used is 3
    pub fn new(step: T, num_iters: Option<u64>) -> PredictorCorrector12<T> {
        let num_iters = num_iters.unwrap_or(3);
        PredictorCorrector12 { step, num_iters }
    }
}

impl<T: Float + ScalarOperand> IVPIntegrator<T> for ExplicitEuler<T> {
    fn step_until(&self, problem: &mut IVProblem<T>, t_final: T) {
        if t_final <= problem.t {
            return;
        }

        // Safety: if this overflows u64, then the number of steps is too large anyway
        // and its okay to panic.
        let num_steps = ((t_final - problem.t) / self.step).ceil().to_u64().unwrap();

        for _ in 0..num_steps {
            problem.y = &problem.y + (problem.rhs)(problem.t, &problem.y) * self.step;
            problem.t = problem.t + self.step;
        }
    }
}

impl<T: Float + ScalarOperand> IVPIntegrator<T> for PredictorCorrector11<T> {
    fn step_until(&self, problem: &mut IVProblem<T>, t_final: T) {
        if t_final <= problem.t {
            return;
        }

        // Safety: if this overflows u64, then the number of steps is too large anyway
        // and its okay to panic.
        let num_steps = ((t_final - problem.t) / self.step).ceil().to_u64().unwrap();
        for _ in 0..num_steps {
            //PECE-mode
            let mut y_guess = &problem.y + (problem.rhs)(problem.t, &problem.y) * self.step;
            for _ in 0..self.num_iters {
                y_guess = &problem.y + (problem.rhs)(problem.t + self.step, &y_guess) * self.step;
            }
            problem.y = &problem.y + (problem.rhs)(problem.t + self.step, &y_guess) * self.step;
            problem.t = problem.t + self.step;
        }
    }
}

impl<T> IVPIntegrator<T> for PredictorCorrector12<T>
where
    T: Float + ScalarOperand,
    f64: Into<T>,
{
    fn step_until(&self, problem: &mut IVProblem<T>, t_final: T) {
        if t_final <= problem.t {
            return;
        }

        // Safety: if this overflows u64, then the number of steps is too large anyway
        // and its okay to panic.
        let num_steps = ((t_final - problem.t) / self.step).ceil().to_u64().unwrap();
        for _ in 0..num_steps {
            //PECE-mode
            let mut y_guess = &problem.y + (problem.rhs)(problem.t, &problem.y) * self.step;
            for _ in 0..self.num_iters {
                y_guess = &problem.y
                    + ((problem.rhs)(problem.t + self.step, &y_guess)
                        + (problem.rhs)(problem.t, &problem.y))
                        / 2.0.into()
                        * self.step;
            }
            problem.y = &problem.y
                + ((problem.rhs)(problem.t + self.step, &y_guess)
                    + (problem.rhs)(problem.t, &problem.y))
                    / 2.0.into()
                    * self.step;
            problem.t = problem.t + self.step;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::StaticVector;
    use ndarray::Array;

    const T_MAX_LOGISTIC: f64 = 0.2;
    const STEP_LOGISTIC: f64 = 0.001;
    const Y0_LOGISTIC: f64 = 0.01;
    const K_CONST_LOGISTIC: f64 = 10.0;

    fn logistic_rhs(_t: f64, y: &StaticVector<f64>) -> StaticVector<f64> {
        let y_diff = -y + 1.0;
        y * y_diff * K_CONST_LOGISTIC
    }

    fn logistic_exact(t: f64) -> f64 {
        let denom = Y0_LOGISTIC + (1.0 - Y0_LOGISTIC) * f64::exp(-K_CONST_LOGISTIC * t);
        Y0_LOGISTIC / denom
    }

    #[test]
    fn explicit_euler_logistic() {
        let mut problem = IVProblem::new(0.0, Array::from_vec(vec![Y0_LOGISTIC]), logistic_rhs);
        let integrator = ExplicitEuler::new(STEP_LOGISTIC);
        integrator.step_until(&mut problem, T_MAX_LOGISTIC);

        let y_result = problem.y[0];
        let y_exact = logistic_exact(problem.t);
        let abs_error = (y_exact - y_result).abs();
        println!("Err explicit {abs_error}");
        assert!(abs_error <= STEP_LOGISTIC * 10.0); // Euler method is first order, so error should be less than 10*step
        assert!((problem.t - T_MAX_LOGISTIC).abs() <= STEP_LOGISTIC);
    }

    #[test]
    fn pc11_logistic() {
        let mut problem = IVProblem::new(0.0, Array::from_vec(vec![Y0_LOGISTIC]), logistic_rhs);
        let integrator = PredictorCorrector11::new(STEP_LOGISTIC, Some(10));
        integrator.step_until(&mut problem, T_MAX_LOGISTIC);

        let y_result = problem.y[0];
        let y_exact = logistic_exact(problem.t);
        let abs_error = (y_exact - y_result).abs();
        println!("Err pc11 {abs_error}");
        assert!(abs_error <= STEP_LOGISTIC * 10.0); // Euler method is first order, so error should be less than 10*step
        assert!((problem.t - T_MAX_LOGISTIC).abs() <= STEP_LOGISTIC);
    }

    #[test]
    fn pc12_logistic() {
        let mut problem = IVProblem::new(0.0, Array::from_vec(vec![Y0_LOGISTIC]), logistic_rhs);
        let integrator = PredictorCorrector12::new(STEP_LOGISTIC, Some(10));
        integrator.step_until(&mut problem, T_MAX_LOGISTIC);

        let y_result = problem.y[0];
        let y_exact = logistic_exact(problem.t);
        let abs_error = (y_exact - y_result).abs();
        println!("Err pc12 {abs_error}");
        assert!(abs_error <= STEP_LOGISTIC); // Crank-Nicolson method is second order, so error should be less than step
        assert!((problem.t - T_MAX_LOGISTIC).abs() <= STEP_LOGISTIC);
    }
}
