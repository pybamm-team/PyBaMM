//! Integrator tuning, one field per diffsol `OdeSolverOptions` knob.
//!
//! Defaults are read from diffsol, except [`SolverOptions::max_nonlinear_solver_failures`].

use diffsol::{OdeEquations, OdeSolverOptions, OdeSolverProblem};

use crate::error::CoreError;

/// Our replacement for diffsol's whole-solve nonlinear-failure budget.
///
/// The counter is cumulative over a solve, so it bounds solve length, not divergence;
/// `min_timestep` catches divergence.
const MAX_NONLINEAR_SOLVER_FAILURES: usize = 100_000;

/// Call `$callback!` with every option, each as `$(#[doc])* name: Type,`.
///
/// The one list that [`SolverOptions`] and each mirror of it, such as the bindings, expand.
#[macro_export]
macro_rules! for_each_solver_option {
    ($callback:ident) => {
        $callback! {
            /// Newton iterations allowed per nonlinear solve.
            max_nonlinear_solver_iterations: usize,
            /// Consecutive step rejections allowed before the solve aborts.
            max_error_test_failures: usize,
            /// Nonlinear-solver convergence failures allowed over the whole solve.
            max_nonlinear_solver_failures: usize,
            /// Newton convergence-test scaling factor.
            nonlinear_solver_tolerance: f64,
            /// Smallest step the controller may take before the solve aborts.
            min_timestep: f64,
            /// Upper bound on step-size growth; `None` keeps diffsol's own.
            max_timestep_growth: Option<f64>,
            /// Lower bound of the step-growth dead zone; `None` keeps diffsol's own.
            min_timestep_growth: Option<f64>,
            /// Upper bound of the step-shrink dead zone; `None` keeps diffsol's own.
            max_timestep_shrink: Option<f64>,
            /// Absolute lower bound on step-size reduction; `None` keeps diffsol's own.
            min_timestep_shrink: Option<f64>,
            /// Steps between linear-solver setups.
            update_jacobian_after_steps: usize,
            /// Steps between full RHS Jacobian re-evaluations.
            update_rhs_jacobian_after_steps: usize,
            /// Relative step-size change that forces a Jacobian update.
            threshold_to_update_jacobian: f64,
            /// Relative step-size change that forces an RHS Jacobian update.
            threshold_to_update_rhs_jacobian: f64,
            /// Proportional gain of the step-size PI controller.
            pi_control_proportional: f64,
            /// Integral gain of the step-size PI controller.
            pi_control_integral: f64,
        }
    };
}

/// Declare [`SolverOptions`] with its copies from and onto diffsol's own options.
macro_rules! define_solver_options {
    ($($(#[$doc:meta])* $field:ident: $ty:ty,)*) => {
        /// Tuning for the diffsol BDF integrator.
        #[derive(Debug, Clone, PartialEq)]
        pub struct SolverOptions {
            $($(#[$doc])* pub $field: $ty,)*
        }

        impl SolverOptions {
            /// diffsol's own defaults, including the failure budget we otherwise raise.
            #[must_use]
            pub fn diffsol_defaults() -> Self {
                let defaults = OdeSolverOptions::<f64>::default();
                Self { $($field: defaults.$field,)* }
            }

            /// Stamp these options onto a freshly built problem, before the solver is created.
            #[must_use]
            pub(crate) const fn apply<Eqn: OdeEquations<T = f64>>(
                &self,
                mut problem: OdeSolverProblem<Eqn>,
            ) -> OdeSolverProblem<Eqn> {
                $(problem.ode_options.$field = self.$field;)*
                problem
            }
        }
    };
}

for_each_solver_option!(define_solver_options);

impl Default for SolverOptions {
    fn default() -> Self {
        Self {
            max_nonlinear_solver_failures: MAX_NONLINEAR_SOLVER_FAILURES,
            ..Self::diffsol_defaults()
        }
    }
}

impl SolverOptions {
    /// Reject values diffsol would take at face value and then misbehave on.
    pub fn validate(&self) -> Result<(), CoreError> {
        // `(name, value, must be positive)`; an absent optional bound is not checked.
        let checks = [
            (
                "nonlinear_solver_tolerance",
                Some(self.nonlinear_solver_tolerance),
                true,
            ),
            ("min_timestep", Some(self.min_timestep), true),
            (
                "threshold_to_update_jacobian",
                Some(self.threshold_to_update_jacobian),
                true,
            ),
            (
                "threshold_to_update_rhs_jacobian",
                Some(self.threshold_to_update_rhs_jacobian),
                true,
            ),
            ("max_timestep_growth", self.max_timestep_growth, true),
            ("min_timestep_growth", self.min_timestep_growth, true),
            ("max_timestep_shrink", self.max_timestep_shrink, true),
            ("min_timestep_shrink", self.min_timestep_shrink, true),
            // A PI gain of zero disables that term, so only finiteness applies.
            (
                "pi_control_proportional",
                Some(self.pi_control_proportional),
                false,
            ),
            ("pi_control_integral", Some(self.pi_control_integral), false),
        ];
        for (name, value, positive) in checks {
            if let Some(got) = value
                && !(got.is_finite() && (!positive || got > 0.0))
            {
                let requirement = if positive { "finite and > 0" } else { "finite" };
                return Err(CoreError::SolverOption {
                    name,
                    requirement,
                    got,
                });
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_defaults_are_valid_and_depart_from_diffsol_only_in_the_failure_budget() {
        let ours = SolverOptions::default();
        ours.validate().expect("defaults invalid");
        // The raised budget clears the measured worst case.
        assert!(ours.max_nonlinear_solver_failures > 10_000);
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
    fn values_diffsol_would_misbehave_on_are_rejected() {
        let tolerance = |got| SolverOptions {
            nonlinear_solver_tolerance: got,
            ..Default::default()
        };
        let positive = ("nonlinear_solver_tolerance", "finite and > 0");
        let cases = [
            (tolerance(0.0), positive),
            (tolerance(-1.0), positive),
            (tolerance(f64::NAN), positive),
            (tolerance(f64::INFINITY), positive),
            // A present optional bound is checked as a required one is.
            (
                SolverOptions {
                    max_timestep_growth: Some(-2.0),
                    ..Default::default()
                },
                ("max_timestep_growth", "finite and > 0"),
            ),
            (
                SolverOptions {
                    pi_control_integral: f64::NAN,
                    ..Default::default()
                },
                ("pi_control_integral", "finite"),
            ),
        ];
        let mut message = String::new();
        for (options, (name, requirement)) in cases {
            let err = options.validate().expect_err(name);
            assert!(
                matches!(err, CoreError::SolverOption { name: n, requirement: r, .. } if n == name && r == requirement),
                "{err:?}"
            );
            message = err.to_string();
        }
        assert_eq!(
            message,
            "solver option pi_control_integral must be finite, got NaN"
        );
    }

    #[test]
    fn an_absent_bound_and_a_zero_pi_gain_are_accepted() {
        // A zero PI gain disables its term rather than breaking the controller.
        for options in [
            SolverOptions {
                max_timestep_growth: None,
                ..Default::default()
            },
            SolverOptions {
                pi_control_proportional: 0.0,
                ..Default::default()
            },
        ] {
            options.validate().expect("accepted");
        }
    }
}
