//! Integrator tuning, one field per diffsol `OdeSolverOptions` knob.
//!
//! Every default is read from diffsol rather than copied, so a caller
//! overriding one knob does not silently inherit a different value for the
//! rest and an upstream change cannot be pinned here unnoticed. The single
//! deliberate departure is [`SolverOptions::max_nonlinear_solver_failures`].

use diffsol::{OdeEquations, OdeSolverOptions, OdeSolverProblem};

use crate::error::CoreError;

/// Our replacement for diffsol's whole-solve nonlinear-failure budget, on the
/// scale of IDAKLU's `max_num_steps`.
///
/// This counter is cumulative over a solve, where diffsol's error-test counter
/// and IDA's `IDASetMaxConvFails` are consecutive per step, so 50 bounds solve
/// *length*, not divergence: a DFN pulse train recovers from ~86 of them per
/// 1800 s. `min_timestep` is what actually catches divergence.
const MAX_NONLINEAR_SOLVER_FAILURES: usize = 100_000;

/// Tuning for the diffsol BDF integrator.
///
/// Field names match diffsol's `OdeSolverOptions` so a knob can be traced from
/// `PyBaMM`'s `options` dict to the integrator without a translation table.
#[derive(Debug, Clone, PartialEq)]
pub struct SolverOptions {
    /// Newton iterations allowed per nonlinear solve.
    pub max_nonlinear_solver_iterations: usize,
    /// Consecutive step rejections allowed before the solve aborts.
    pub max_error_test_failures: usize,
    /// Nonlinear-solver convergence failures allowed over the whole solve.
    pub max_nonlinear_solver_failures: usize,
    /// Newton convergence-test scaling factor.
    pub nonlinear_solver_tolerance: f64,
    /// Smallest step the controller may take before the solve aborts.
    pub min_timestep: f64,
    /// Upper bound on step-size growth; `None` keeps diffsol's own.
    pub max_timestep_growth: Option<f64>,
    /// Lower bound of the step-growth dead zone; `None` keeps diffsol's own.
    pub min_timestep_growth: Option<f64>,
    /// Upper bound of the step-shrink dead zone; `None` keeps diffsol's own.
    pub max_timestep_shrink: Option<f64>,
    /// Absolute lower bound on step-size reduction; `None` keeps diffsol's own.
    pub min_timestep_shrink: Option<f64>,
    /// Steps between linear-solver setups.
    pub update_jacobian_after_steps: usize,
    /// Steps between full RHS Jacobian re-evaluations.
    pub update_rhs_jacobian_after_steps: usize,
    /// Relative step-size change that forces a Jacobian update.
    pub threshold_to_update_jacobian: f64,
    /// Relative step-size change that forces an RHS Jacobian update.
    pub threshold_to_update_rhs_jacobian: f64,
    /// PI step controller proportional gain.
    pub pi_control_proportional: f64,
    /// PI step controller integral gain.
    pub pi_control_integral: f64,
}

impl Default for SolverOptions {
    fn default() -> Self {
        Self {
            max_nonlinear_solver_failures: MAX_NONLINEAR_SOLVER_FAILURES,
            ..Self::diffsol_defaults()
        }
    }
}

impl SolverOptions {
    /// diffsol's own defaults, including the failure budget we otherwise raise.
    ///
    /// Read straight off `OdeSolverOptions` so a bump changes this with it.
    #[must_use]
    pub fn diffsol_defaults() -> Self {
        let defaults = OdeSolverOptions::<f64>::default();
        Self {
            max_nonlinear_solver_iterations: defaults.max_nonlinear_solver_iterations,
            max_error_test_failures: defaults.max_error_test_failures,
            max_nonlinear_solver_failures: defaults.max_nonlinear_solver_failures,
            nonlinear_solver_tolerance: defaults.nonlinear_solver_tolerance,
            min_timestep: defaults.min_timestep,
            max_timestep_growth: defaults.max_timestep_growth,
            min_timestep_growth: defaults.min_timestep_growth,
            max_timestep_shrink: defaults.max_timestep_shrink,
            min_timestep_shrink: defaults.min_timestep_shrink,
            update_jacobian_after_steps: defaults.update_jacobian_after_steps,
            update_rhs_jacobian_after_steps: defaults.update_rhs_jacobian_after_steps,
            threshold_to_update_jacobian: defaults.threshold_to_update_jacobian,
            threshold_to_update_rhs_jacobian: defaults.threshold_to_update_rhs_jacobian,
            pi_control_proportional: defaults.pi_control_proportional,
            pi_control_integral: defaults.pi_control_integral,
        }
    }

    /// Reject values diffsol would take at face value and then misbehave on.
    ///
    /// Counts are unsigned, so only the floating-point knobs need checking: all
    /// must be finite, and those acting as a scale or a bound must be positive.
    ///
    /// # Errors
    ///
    /// [`CoreError::SolverOption`] naming the first offending field.
    pub fn validate(&self) -> Result<(), CoreError> {
        let positive = [
            (
                "nonlinear_solver_tolerance",
                self.nonlinear_solver_tolerance,
            ),
            ("min_timestep", self.min_timestep),
            (
                "threshold_to_update_jacobian",
                self.threshold_to_update_jacobian,
            ),
            (
                "threshold_to_update_rhs_jacobian",
                self.threshold_to_update_rhs_jacobian,
            ),
        ];
        for (name, value) in positive {
            if !value.is_finite() || value <= 0.0 {
                return Err(CoreError::SolverOption {
                    name: name.to_string(),
                    got: value,
                });
            }
        }

        let optional = [
            ("max_timestep_growth", self.max_timestep_growth),
            ("min_timestep_growth", self.min_timestep_growth),
            ("max_timestep_shrink", self.max_timestep_shrink),
            ("min_timestep_shrink", self.min_timestep_shrink),
        ];
        for (name, value) in optional {
            if let Some(value) = value
                && (!value.is_finite() || value <= 0.0)
            {
                return Err(CoreError::SolverOption {
                    name: name.to_string(),
                    got: value,
                });
            }
        }

        // A PI gain of zero disables that term, so only finiteness applies.
        let finite = [
            ("pi_control_proportional", self.pi_control_proportional),
            ("pi_control_integral", self.pi_control_integral),
        ];
        for (name, value) in finite {
            if !value.is_finite() {
                return Err(CoreError::SolverOption {
                    name: name.to_string(),
                    got: value,
                });
            }
        }

        Ok(())
    }

    /// Stamp these options onto a freshly built problem.
    ///
    /// `OdeBuilder` exposes no setters for them, so the problem's public
    /// `ode_options` is the wiring point; it is read when the solver is created.
    #[must_use]
    pub const fn apply<Eqn: OdeEquations<T = f64>>(
        &self,
        mut problem: OdeSolverProblem<Eqn>,
    ) -> OdeSolverProblem<Eqn> {
        let options = &mut problem.ode_options;
        options.max_nonlinear_solver_iterations = self.max_nonlinear_solver_iterations;
        options.max_error_test_failures = self.max_error_test_failures;
        options.max_nonlinear_solver_failures = self.max_nonlinear_solver_failures;
        options.nonlinear_solver_tolerance = self.nonlinear_solver_tolerance;
        options.min_timestep = self.min_timestep;
        options.max_timestep_growth = self.max_timestep_growth;
        options.min_timestep_growth = self.min_timestep_growth;
        options.max_timestep_shrink = self.max_timestep_shrink;
        options.min_timestep_shrink = self.min_timestep_shrink;
        options.update_jacobian_after_steps = self.update_jacobian_after_steps;
        options.update_rhs_jacobian_after_steps = self.update_rhs_jacobian_after_steps;
        options.threshold_to_update_jacobian = self.threshold_to_update_jacobian;
        options.threshold_to_update_rhs_jacobian = self.threshold_to_update_rhs_jacobian;
        options.pi_control_proportional = self.pi_control_proportional;
        options.pi_control_integral = self.pi_control_integral;
        problem
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_the_failure_budget_departs_from_diffsol() {
        let ours = SolverOptions::default();
        let theirs = SolverOptions::diffsol_defaults();
        assert_ne!(
            ours.max_nonlinear_solver_failures,
            theirs.max_nonlinear_solver_failures
        );
        assert_eq!(
            SolverOptions {
                max_nonlinear_solver_failures: theirs.max_nonlinear_solver_failures,
                ..ours
            },
            theirs,
            "a second departure from diffsol's defaults needs its own rationale",
        );
    }

    #[test]
    fn the_raised_budget_clears_the_measured_worst_case() {
        // DFN pulse_train spends ~141 with sensitivities on; a default that
        // merely doubled diffsol's 50 would still fail it.
        assert!(SolverOptions::default().max_nonlinear_solver_failures > 10_000);
    }

    #[test]
    fn defaults_validate() {
        SolverOptions::default()
            .validate()
            .expect("defaults invalid");
    }

    #[test]
    fn non_positive_and_non_finite_scales_are_rejected() {
        for bad in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            let options = SolverOptions {
                nonlinear_solver_tolerance: bad,
                ..Default::default()
            };
            let Err(err) = options.validate() else {
                panic!("{bad} should have been rejected");
            };
            assert!(matches!(err, CoreError::SolverOption { ref name, .. }
                if name == "nonlinear_solver_tolerance"));
        }
    }

    #[test]
    fn an_absent_optional_bound_is_not_validated() {
        let options = SolverOptions {
            max_timestep_growth: None,
            ..Default::default()
        };
        options.validate().expect("None must be accepted");
    }

    #[test]
    fn a_present_optional_bound_is_validated() {
        let options = SolverOptions {
            max_timestep_growth: Some(-2.0),
            ..Default::default()
        };
        let err = options.validate().expect_err("negative growth accepted");
        assert!(matches!(err, CoreError::SolverOption { ref name, .. }
            if name == "max_timestep_growth"));
    }

    #[test]
    fn a_zero_pi_gain_is_accepted_because_it_disables_the_term() {
        let options = SolverOptions {
            pi_control_proportional: 0.0,
            ..Default::default()
        };
        options.validate().expect("zero gain must be accepted");
    }

    #[test]
    fn a_non_finite_pi_gain_is_rejected() {
        let options = SolverOptions {
            pi_control_integral: f64::NAN,
            ..Default::default()
        };
        let err = options.validate().expect_err("NaN gain accepted");
        assert!(matches!(err, CoreError::SolverOption { ref name, .. }
            if name == "pi_control_integral"));
    }
}
