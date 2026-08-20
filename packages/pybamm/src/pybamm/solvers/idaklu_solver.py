# mypy: ignore-errors
import logging
import numbers
import warnings
from enum import IntEnum

import casadi
import numpy as np
from pybammsolvers import idaklu
from scipy.sparse.linalg import spsolve

import pybamm
from pybamm.codegen.compilation import aot_compile
from pybamm.solvers.base_solver import (
    flatten_inputs,
    stack_inputs,
    validate_rust_sensitivity_widths,
)
from pybamm.solvers.observation import (
    NativeInterpolatingObservation,
    OutputAssembly,
)
from pybamm.solvers.rust_lowering import RustModelLowering

_UNSET = object()


def _sensitivity_scales(inputs_dict, sensitivity_names):
    """IDAS ``pbar``: the magnitude of each differentiated parameter.

    IDAS weights the *scaled* sensitivity ``pbar_i * dy/dp_i`` like a state, so
    ``pbar_i = |p_i|`` makes a column of ``dy/dp`` comparable to ``y`` however
    small the parameter is. Left at the default 1.0, a diffusivity around
    1e-15 produces a sensitivity column around 1e14 that no absolute tolerance
    can accommodate, and the corrector fails to converge.

    Magnitudes are handed over raw; the solver clamps the zero and non-finite
    cases IDAS rejects. A vector-valued input takes one scale for the block.

    Parameters
    ----------
    inputs_dict : dict
        Input values for one input set.
    sensitivity_names : list of str
        Differentiated parameter names, in the solver's column order.

    Returns
    -------
    :class:`numpy.ndarray`
        One scale per differentiated parameter, ``(len(sensitivity_names),)``.
    """
    return np.asarray(
        [
            np.abs(np.asarray(inputs_dict[name], dtype=np.float64)).max()
            for name in sensitivity_names
        ],
        dtype=np.float64,
    )


# Mirrors SUNDIALS ``IDA_ROOT_RETURN`` in ``sundials/include/ida/ida.h``.
# Returned by ``IDASolve`` (and surfaced via ``Solution.flag``) when the
# integrator has located one or more root function zeros.
_IDA_ROOT_RETURN = 2

# Function entries staged in ``_setup`` while building the C++ solver
# group; popped after construction (C++ owns them via shared_ptr).
_SETUP_FCN_KEYS = (
    "rhs_algebraic",
    "jac_times_cjmass",
    "jac_rhs_algebraic_action",
    "mass_action",
    "sensfn",
    "rootfn",
    "alg_res",
    "alg_jac",
)
_SETUP_FCN_LIST_KEYS = (
    "var_idaklu_fcns",
    "dvar_dy_idaklu_fcns",
    "dvar_dp_idaklu_fcns",
)

# Attributes holding casadi.Function graphs (or the C++ solver) that are
# rebuilt from the model on the next solve(). Dropped in __getstate__ so a
# pickled solver doesn't carry these heavy, non-portable objects.
_REBUILDABLE_STATE_KEYS = (
    "_setup",
    "_model_set_up",
    "computed_var_fcns",
    "computed_dvar_dy_fcns",
    "computed_dvar_dp_fcns",
    "_time_integral_vars",
)


class IDAKLUSolver(pybamm.BaseSolver):
    """
    Solve a discretised model, using sundials with the KLU sparse linear solver.

    Parameters
    ----------
    rtol : float, optional
        The relative tolerance for the solver (default is 1e-4).
    atol : float or :class:`numpy.ndarray`, optional
        The absolute tolerance for the solver, either shared by every state or
        one entry per state (default is 1e-6).
    root_method : str or pybamm algebraic solver class, optional
        The method to use to find initial conditions (for DAE solvers).
        Default is None, which uses a custom Newton solver for consistent
        initial conditions (recommended). If "casadi", the solver uses
        casadi's Newton rootfinding algorithm as a fallback.
        Otherwise, the solver uses 'scipy.optimize.root' with method
        specified by 'root_method' (e.g. "lm", "hybr", ...)
    root_tol : float, optional
        The tolerance for the initial-condition solver (default is 1e-6).
    extrap_tol : float, optional
        The tolerance to assert whether extrapolation occurs or not (default is 0).
    on_extrapolation : str, optional
        What to do if the solver is extrapolating. Options are "warn", "error", or "ignore".
        Default is "warn".
    on_failure : str, optional
        What to do if a solver error flag occurs. Options are "warn", "error", or "ignore".
        Default is "error".
    output_variables : list[str], optional
        List of variables to calculate and return. If none are specified then
        the complete state vector is returned (can be very large) (default is [])
    store_first_last : bool, optional
        If True, only the first and last sample of each integration window are
        stored (one experiment step in :meth:`pybamm.Simulation.solve`, or the
        full ``[t_eval[0], t_eval[-1]]`` window in :meth:`solve`). Intended for
        memory-light long experiments whose post-processing only reads per-step
        first/last values. Note: with this flag on, IDAKLU's Hermite
        interpolation is disabled (see :attr:`options["hermite_interpolation"]`)
        and any query at a non-endpoint time within a step falls back to linear
        interpolation across the whole step, so this flag is **not**
        appropriate when post-processing queries an intra-step time.
        Default is False.
    options: dict, optional
        Addititional options to pass to the solver, by default:

        .. code-block:: python

            options = {
                # Print statistics of the solver after every solve
                "print_stats": False,
                # If ``True``, ahead-of-time compile each casadi ``Function``
                # to a shared library via the system C compiler (see
                # :mod:`pybamm.codegen.compilation`). If ``False`` (default)
                # the casadi in-process virtual machine is used.
                "compile": False,
                # Number of threads available for OpenMP (must be greater than or equal to `num_solvers`)
                "num_threads": 1,
                # Number of solvers to use in parallel (for solving multiple sets of input parameters in parallel)
                "num_solvers": num_threads,
                ## Linear solver interface
                # name of sundials linear solver to use options are: "SUNLinSol_KLU",
                # "SUNLinSol_Dense", "SUNLinSol_Band", "SUNLinSol_SPBCGS",
                # "SUNLinSol_SPFGMR", "SUNLinSol_SPGMR", "SUNLinSol_SPTFQMR",
                "linear_solver": "SUNLinSol_KLU",
                # Jacobian form, can be "none", "dense",
                # "banded", "sparse", "matrix-free"
                "jacobian": "sparse",
                # Preconditioner for iterative solvers, can be "none", "BBDP"
                "preconditioner": "BBDP",
                # For iterative linear solver preconditioner, bandwidth of
                # approximate jacobian
                "precon_half_bandwidth": 5,
                # For iterative linear solver preconditioner, bandwidth of
                # approximate jacobian that is kept
                "precon_half_bandwidth_keep": 5,
                # For iterative linear solvers, max number of iterations
                "linsol_max_iterations": 5,
                # Ratio between linear and nonlinear tolerances
                "epsilon_linear_tolerance": 0.05,
                # Increment factor used in DQ Jacobian-vector product approximation
                "increment_factor": 1.0,
                # Enable or disable linear solution scaling
                "linear_solution_scaling": True,
                # Silence Sundials errors during solve
                "silence_sundials_errors": False,
                ## Main solver
                # Maximum order of the linear multistep method
                "max_order_bdf": 5,
                # Maximum number of steps to be taken by the solver in its attempt to
                # reach the next output time.
                # Note: this value differs from the IDA default of 500
                "max_num_steps": 100000,
                # Initial step size. The solver default is used if this is left at 0.0
                "dt_init": 0.0,
                # Minimum absolute step size. The solver default is used if this is
                # left at 0.0
                "dt_min": 0.0,
                # Maximum absolute step size. The solver default is used if this is
                # left at 0.0
                "dt_max": 0.0,
                # Maximum number of error test failures in attempting one step
                "max_error_test_failures": 10,
                # Maximum number of nonlinear solver iterations at one step
                # Note: this value differs from the IDA default of 4
                "max_nonlinear_iterations": 40,
                # Maximum number of nonlinear solver convergence failures at one step
                # Note: this value differs from the IDA default of 10
                "max_convergence_failures": 100,
                # Safety factor in the nonlinear convergence test
                "nonlinear_convergence_coefficient": 0.33,
                # Suppress algebraic variables from error test
                "suppress_algebraic_error": False,
                # Store Hermite interpolation data for the solution.
                # Note: this option is always disabled if output_variables are given
                # or if t_interp values are specified
                "hermite_interpolation": True,
                # Setting hermite_reduction_factor > 1.0 compresses the solution size
                # by introducing a small amount of error to the Hermite spline
                # interpolant. A value of `2.0` roughly corresponds to a maximum 2x
                # increase in error (practically the error is much smaller), while
                # reducing the number of saved states by around 5-6x. This option is
                # only active if `hermite_interpolation` is True and sensitivities
                # are disabled.
                "hermite_reduction_factor": 1.0,
                ## Initial conditions calculation
                # Positive constant in the Newton iteration convergence test within the
                # initial condition calculation
                "nonlinear_convergence_coefficient_ic": 0.0033,
                # Maximum number of steps allowed when `init_all_y_ic = False`
                # Note: this value differs from the IDA default of 5
                "max_num_steps_ic": 50,
                # Maximum number of the approximate Jacobian or preconditioner evaluations
                # allowed when the Newton iteration appears to be slowly converging
                # Note: this value differs from the IDA default of 4
                "max_num_jacobians_ic": 40,
                # Maximum number of Newton iterations allowed in any one attempt to solve
                # the initial conditions calculation problem
                # Note: this value differs from the IDA default of 10
                "max_num_iterations_ic": 100,
                # Maximum number of linesearch backtracks allowed in any Newton iteration,
                # when solving the initial conditions calculation problem
                "max_linesearch_backtracks_ic": 5,
                # Turn off linesearch
                "linesearch_off_ic": False,
                # How to calculate the initial conditions.
                # "True": calculate all y0 given ydot0
                # "False": calculate y_alg0 and ydot_diff0 given y_diff0
                "init_all_y_ic": False,
                # Internally calculate consistent initial conditions
                "calc_ic": True,
                ## Newton IC solver
                # "auto": use dedicated sub-block solver when possible.
                #         This can result in a potentially smaller system of only the
                #         algebraic variables. Requires a direct linear solver and
                #         a standard form DAE.
                # "full": always use IDA's full-system linear solve. This uses the full
                #         system of equationd and can handle non-standard form DAEs and
                #         all classes of linear solvers/
                "newton_mode": "auto",
                ## Early termination
                # Maximum number of consecutive steps allowed without advancing
                # the solution time by at least `t_no_progress` seconds.
                # If set to 0, this feature is disabled.
                "num_steps_no_progress": 0,
                # Minimum required time advancement (in seconds) after
                # `num_steps_no_progress` consecutive steps.
                # If set to 0.0, this feature is disabled.
                "t_no_progress": 0.0,
            }

        Note: Options apply to both 'casadi' and 'rust' formats except where noted.


    """

    _integrates_via_compiled_model = True

    # idaklu observation is native (Rust): full-state values via native
    # cubic-Hermite, sensitivities via the native chain rule.
    _observes_via_compiled_model = True

    class StateID(IntEnum):
        ALGEBRAIC = 0
        DIFFERENTIAL = 1

    def __init__(
        self,
        rtol=1e-4,
        atol=1e-6,
        root_method=_UNSET,
        root_tol=1e-6,
        extrap_tol=None,
        on_extrapolation=None,
        output_variables=None,
        on_failure=None,
        options=None,
        store_first_last=False,
    ):
        self.output_variables = [] if output_variables is None else output_variables

        self._options = self._combine_options(options)

        # By default, we use an internal nonlinear solver within pybammsolvers
        # to compute the initial conditions. As a fallback, we use python bindings
        # for the same solver
        if root_method is _UNSET:
            root_method = None if self._internal_initialisation else "nonlinear_solver"

        super().__init__(
            method="ida",
            rtol=rtol,
            atol=atol,
            root_method=root_method,
            root_tol=root_tol,
            extrap_tol=extrap_tol,
            output_variables=output_variables,
            on_extrapolation=on_extrapolation,
            on_failure=on_failure,
            store_first_last=store_first_last,
        )
        self.name = "IDA KLU solver"
        self._supports_interp = True
        self._supports_t_eval_discontinuities = True

        pybamm.citations.register("Hindmarsh2000")
        pybamm.citations.register("Hindmarsh2005")

    def _validate_rust_compatibility(self, model, inputs):
        """Runtime guards for unsupported Rust backend features."""
        if not model.use_jacobian:
            raise pybamm.SolverError("KLU requires the Jacobian")
        # Equality makes SetupOptions derive one thread per solver, so nobody
        # gets OpenMP N_Vectors: a pessimisation, and not run-to-run reproducible.
        num_threads = self._options["num_threads"]
        if self._options.get("num_solvers", num_threads) != num_threads:
            pybamm.logger.warning(
                "The Rust backend runs one solver per thread; num_solvers is "
                f"set to num_threads ({num_threads})."
            )
        self._options["num_solvers"] = num_threads
        if model.calculate_sensitivities:
            validate_rust_sensitivity_widths(model, model.calculate_sensitivities)

    def _is_identity_matrix(self, matrix):
        """Check if sparse matrix is identity."""
        import scipy.sparse as sp

        if not sp.issparse(matrix):
            return False
        n = matrix.shape[0]
        if matrix.shape[0] != matrix.shape[1]:
            return False
        if matrix.nnz != n:
            return False
        # Check diagonal is all ones
        diag = matrix.diagonal()
        return np.allclose(diag, 1.0)

    def _combine_options(self, user_options: dict | None) -> dict:
        user_options = user_options or {}
        num_solvers = user_options.get("num_threads", 1)

        default_options = {
            "print_stats": False,
            "compile": False,
            "jacobian": "sparse",
            "preconditioner": "BBDP",
            "precon_half_bandwidth": 5,
            "precon_half_bandwidth_keep": 5,
            "num_threads": 1,
            "num_solvers": num_solvers,
            "linear_solver": "SUNLinSol_KLU",
            "linsol_max_iterations": 5,
            "epsilon_linear_tolerance": 0.05,
            "increment_factor": 1.0,
            "linear_solution_scaling": True,
            "silence_sundials_errors": False,
            "max_order_bdf": 5,
            "max_num_steps": 100000,
            "dt_init": 0.0,
            "dt_min": 0.0,
            "dt_max": 0.0,
            "max_error_test_failures": 10,
            "max_nonlinear_iterations": 40,
            "max_convergence_failures": 100,
            "nonlinear_convergence_coefficient": 0.33,
            "suppress_algebraic_error": False,
            "hermite_interpolation": True,
            "hermite_reduction_factor": 1.0,
            "nonlinear_convergence_coefficient_ic": 0.0033,
            "max_num_steps_ic": 50,
            "max_num_jacobians_ic": 40,
            "max_num_iterations_ic": 100,
            "max_linesearch_backtracks_ic": 5,
            "linesearch_off_ic": False,
            "init_all_y_ic": False,
            "calc_ic": True,
            "newton_step_tol": 1e-4,
            "newton_mode": "auto",
            "num_steps_no_progress": 0,
            "t_no_progress": 0.0,
        }
        options = self._overlay_options(
            default_options, user_options, solver_name="IDAKLU"
        )
        self._check_options(options)
        return options

    def _check_options(self, options: dict):
        hermite_reduction_factor = options["hermite_reduction_factor"]
        if hermite_reduction_factor > 1.0:
            if self.output_variables:
                raise pybamm.SolverError(
                    "hermite_reduction_factor cannot be used with "
                    "output_variables. Both are memory-saving options "
                    "that are mutually exclusive."
                )
            if not options["hermite_interpolation"]:
                raise pybamm.SolverError(
                    "hermite_reduction_factor requires "
                    "hermite_interpolation to be enabled."
                )
        else:
            if hermite_reduction_factor < 1.0:
                raise pybamm.SolverError("hermite_reduction_factor must be >= 1.0.")

        if not isinstance(options["compile"], bool):
            raise pybamm.SolverError("compile must be a bool")

    def _set_up_rust(self, model, inputs_dict, y0):
        """Set up solver with Rust evaluation backend."""
        lowering = RustModelLowering(model, inputs_dict)
        lowering.state_residual()

        _, sensitivity_names = lowering.sensitivity_indices(
            model.calculate_sensitivities
        )

        if self._options["hermite_reduction_factor"] > 1.0 and sensitivity_names:
            warnings.warn(
                "Setting hermite_reduction_factor > 1.0 is not currently supported "
                "with sensitivities. The hermite_reduction_factor option will be "
                "ignored.",
                pybamm.SolverWarning,
                stacklevel=2,
            )

        # A time-integral output lowers to its integrand trajectory; the postfix
        # sum runs post-solve.
        _, output_lens = lowering.outputs(
            self.output_variables, time_integral_vars=self._time_integral_vars
        )

        lowering.algebraic_block()
        lowering.termination_events()

        n_inputs = lowering.n_inputs
        rust_model = lowering.compile()
        lowering.bind_generic_evaluators(rust_model)

        # Get sparsity pattern (CSC format)
        colptrs, rowinds = rust_model.csc_sparsity_pattern()
        nnz = rust_model.nnz
        # Mirror the existing CasADi path: only expose algebraic-subblock
        # callbacks when Newton IC mode is "auto".
        enable_alg_subblock = (
            self._options.get("newton_mode", "auto") == "auto"
            and rust_model.has_algebraic
        )
        if enable_alg_subblock:
            alg_jac_rows, alg_jac_cols = (
                rust_model.algebraic_jacobian_sparsity_pattern()
            )
            alg_jac_nnz = rust_model.algebraic_jacobian_nnz
            n_alg = rust_model.n_algebraic
        else:
            alg_jac_rows = np.array([], dtype=np.int64)
            alg_jac_cols = np.array([], dtype=np.int64)
            alg_jac_nnz = 0
            n_alg = 0

        # IDA `id` vector: 1.0 for differential, 0.0 for algebraic.
        # Inferred from the mass matrix diagonal by the Rust core.
        ids = np.asarray(rust_model.algebraic_ids(), dtype=np.float64)

        atol = self._check_atol_type(getattr(model, "atol", self.atol), model)

        # Store for solver creation
        self._setup = {
            "rust_model": rust_model,
            # One evaluator per parallel solver, all sharing rust_model's tape.
            "rust_evaluators": rust_model.evaluator_pool(self._options["num_solvers"]),
            "number_of_states": len(y0),
            "number_of_inputs": n_inputs,
            "number_of_sensitivity_parameters": rust_model.n_sens_params,
            "sensitivity_names": sensitivity_names,
            "output_lens": output_lens,
            "output_assembly": OutputAssembly(
                self.output_variables,
                output_lens,
                time_integrals=self._time_integral_vars,
            ),
            "jac_colptrs": np.asarray(colptrs, dtype=np.int64),
            "jac_rowvals": np.asarray(rowinds, dtype=np.int64),
            "jac_nnz": nnz,
            "n_alg": n_alg,
            "alg_jac_rowvals": np.asarray(alg_jac_rows, dtype=np.int64),
            "alg_jac_colvals": np.asarray(alg_jac_cols, dtype=np.int64),
            "alg_jac_nnz": alg_jac_nnz,
            "ids": ids,
            "atol": atol,
            # `num_of_events` mirrors the CasADi setup key so the shared
            # `_post_process_solution` can read the event count uniformly.
            "num_of_events": rust_model.n_events,
            # Observation tapes cached 1:1 with the rust model; recreated on every
            # `set_up`, so it can never serve tapes from a stale graph.
            "rust_observation_cache": {},
        }

        self._setup["solver"] = idaklu.create_rust_solver_group(
            rust_evaluators=self._setup["rust_evaluators"],
            number_of_states=self._setup["number_of_states"],
            number_of_inputs=self._setup["number_of_inputs"],
            number_of_sens_params=self._setup["number_of_sensitivity_parameters"],
            number_of_events=self._setup["num_of_events"],
            output_lens=self._setup["output_lens"],
            jac_colptrs=self._setup["jac_colptrs"],
            jac_rowvals=self._setup["jac_rowvals"],
            jac_nnz=self._setup["jac_nnz"],
            n_alg=self._setup["n_alg"],
            alg_jac_rowvals=self._setup["alg_jac_rowvals"],
            alg_jac_colvals=self._setup["alg_jac_colvals"],
            alg_jac_nnz=self._setup["alg_jac_nnz"],
            rhs_alg_id=self._setup["ids"],
            atol=self._setup["atol"],
            rtol=self.rtol,
            options=self._options,
        )

    def set_up(self, model, inputs=None, t_eval=None, ics_only=False):
        if model.convert_to_format not in ("casadi", "rust"):
            pybamm.logger.warning(
                f"Converting {model.name} to CasADi for solving with IDAKLUSolver"
            )
            model.convert_to_format = "casadi"
        base_set_up_return = super().set_up(model, inputs, t_eval, ics_only)

        if isinstance(inputs, list):
            # for setup, just use first input dict
            inputs_dict = inputs[0]
        else:
            inputs_dict = inputs or {}
        # stack inputs
        if inputs_dict:
            arrays_to_stack = [np.array(x).reshape(-1, 1) for x in inputs_dict.values()]
            stacked_inputs = np.vstack(arrays_to_stack)
        else:
            stacked_inputs = np.array([[]])

        y0 = model.y0_list[0]

        if isinstance(y0, casadi.DM):
            y0 = y0.full()
        y0 = y0.flatten()

        if ics_only:
            return base_set_up_return

        if model.convert_to_format == "rust":
            self._validate_rust_compatibility(model, inputs_dict)
            self._set_up_rust(model, inputs_dict, y0)
            return base_set_up_return

        if model.convert_to_format != "casadi":
            msg = "The python-idaklu solver has been deprecated."
            warnings.warn(msg, DeprecationWarning, stacklevel=2)
            raise pybamm.SolverError(
                f"Unsupported option for convert_to_format={model.convert_to_format}"
            )

        mass_matrix = model.mass_matrix.entries

        # construct residuals function by binding inputs
        # TODO: do we need densify here?
        rhs_algebraic = model.rhs_algebraic_eval

        if not model.use_jacobian:
            raise pybamm.SolverError("KLU requires the Jacobian")

        # need to provide jacobian_rhs_alg - cj * mass_matrix
        t_casadi = casadi.MX.sym("t")
        y_casadi = casadi.MX.sym("y", model.len_rhs_and_alg)
        cj_casadi = casadi.MX.sym("cj")
        p_casadi = {}
        for name, value in inputs_dict.items():
            if isinstance(value, numbers.Number):
                p_casadi[name] = casadi.MX.sym(name)
            else:
                p_casadi[name] = casadi.MX.sym(name, value.shape[0])
        p_casadi_stacked = casadi.vertcat(*[p for p in p_casadi.values()])

        jac_times_cjmass_expression = (
            model.jac_rhs_algebraic_eval(t_casadi, y_casadi, p_casadi_stacked)
            - cj_casadi * mass_matrix
        )
        if self._options["jacobian"] == "dense":
            jac_times_cjmass_expression = casadi.densify(jac_times_cjmass_expression)

        jac_times_cjmass = casadi.Function(
            "jac_times_cjmass",
            [t_casadi, y_casadi, p_casadi_stacked, cj_casadi],
            [jac_times_cjmass_expression],
        )

        jac_times_cjmass_sparsity = jac_times_cjmass.sparsity_out(0)
        jac_bw_lower = jac_times_cjmass_sparsity.bw_lower()
        jac_bw_upper = jac_times_cjmass_sparsity.bw_upper()
        jac_times_cjmass_nnz = jac_times_cjmass_sparsity.nnz()
        jac_times_cjmass_colptrs = np.array(
            jac_times_cjmass_sparsity.colind(), dtype=np.int64
        )
        jac_times_cjmass_rowvals = np.array(
            jac_times_cjmass_sparsity.row(), dtype=np.int64
        )

        v_casadi = casadi.MX.sym("v", model.len_rhs_and_alg)

        jac_rhs_algebraic_action = model.jac_rhs_algebraic_action_eval

        if model.is_standard_form_dae:
            # index v_casadi at the first model.len_rhs entries
            # concatenate it with zeros at the end
            mass_action_expression = casadi.vertcat(
                v_casadi[: model.len_rhs],
                np.zeros(model.len_alg),
            )
        else:
            mass_action_expression = casadi.densify(casadi.DM(mass_matrix) @ v_casadi)

        # also need the action of the mass matrix on a vector
        mass_action = casadi.Function(
            "mass_action", [v_casadi], [mass_action_expression]
        )

        num_of_events = len(model.terminate_events_eval)

        # rootfn needs to return an array of length num_of_events
        rootfn = casadi.Function(
            "rootfn",
            [t_casadi, y_casadi, p_casadi_stacked],
            [
                casadi.vertcat(
                    *[
                        event(t_casadi, y_casadi, p_casadi_stacked)
                        for event in model.terminate_events_eval
                    ]
                )
            ],
        )

        # get ids of rhs and algebraic variables
        ids = np.concatenate(
            (
                np.full(model.len_rhs, self.StateID.DIFFERENTIAL, dtype=np.int64),
                np.full(model.len_alg, self.StateID.ALGEBRAIC, dtype=np.int64),
            )
        )

        if model.jacp_rhs_algebraic_eval is not None:
            sensitivity_names = model.calculate_sensitivities
            if model.convert_to_format == "casadi":
                number_of_sensitivity_parameters = model.jacp_rhs_algebraic_eval.n_out()
            else:
                number_of_sensitivity_parameters = len(sensitivity_names)
        else:
            number_of_sensitivity_parameters = 0
            sensitivity_names = []

        # for the casadi solver we just give it dFdp_i
        if model.jacp_rhs_algebraic_eval is None:
            sensfn = casadi.Function("sensfn", [], [])
        else:
            sensfn = model.jacp_rhs_algebraic_eval

        atol = getattr(model, "atol", self.atol)
        atol = self._check_atol_type(atol, model)

        # Build algebraic-only residual and Jacobian for Newton sub-block mode.
        # When newton_mode="full", skip these so the C++ solver uses the
        # full-system IDA linear solve (DECOUPLED_FULL or COUPLED_FULL),
        # which supports any linear solver including iterative ones.
        if self._options.get("newton_mode", "auto") == "auto":
            alg_res_fn = model.algebraic_eval
            jac_alg_fn = model.jac_algebraic_eval
        else:
            alg_res_fn = casadi.Function("empty_alg_res", [], [])
            jac_alg_fn = casadi.Function("empty_alg_jac", [], [])

        if (
            self._options["hermite_reduction_factor"] > 1.0
            and number_of_sensitivity_parameters > 0
        ):
            warnings.warn(
                "Setting hermite_reduction_factor > 1.0 is not currently supported "
                "with sensitivities. The hermite_reduction_factor option will be "
                "ignored.",
                pybamm.SolverWarning,
                stacklevel=2,
            )

        # Collect all casadi functions for AOT compilation. With compile=True,
        # all functions are bundled into a single shared library via one gcc
        # invocation; each serialized Function then points at its entry point.
        has_sens = (len(stacked_inputs) > 0) and model.calculate_sensitivities
        fns = dict(
            zip(
                _SETUP_FCN_KEYS,
                (
                    rhs_algebraic,
                    jac_times_cjmass,
                    jac_rhs_algebraic_action,
                    mass_action,
                    sensfn,
                    rootfn,
                    alg_res_fn,
                    jac_alg_fn,
                ),
                strict=True,
            )
        )
        for key in self.output_variables:
            fns[f"var:{key}"] = self.computed_var_fcns[key]
            if has_sens:
                fns[f"dvar_dy:{key}"] = self.computed_dvar_dy_fcns[key]
                fns[f"dvar_dp:{key}"] = self.computed_dvar_dp_fcns[key]

        if self._options["compile"]:
            compiled = aot_compile(list(fns.values()))
            fns = dict(zip(fns.keys(), compiled, strict=True))

        def to_idaklu(fn):
            return idaklu.generate_function(fn.serialize())

        self._setup = {
            "number_of_states": len(y0),
            "inputs": len(stacked_inputs),
            "jac_bandwidth_upper": jac_bw_upper,
            "jac_bandwidth_lower": jac_bw_lower,
            "jac_times_cjmass_colptrs": jac_times_cjmass_colptrs,
            "jac_times_cjmass_rowvals": jac_times_cjmass_rowvals,
            "jac_times_cjmass_nnz": jac_times_cjmass_nnz,
            "num_of_events": num_of_events,
            "ids": ids,
            "atol": atol,
            "sensitivity_names": sensitivity_names,
            "number_of_sensitivity_parameters": number_of_sensitivity_parameters,
            "standard_form_dae": model.is_standard_form_dae,
            "output_variables": self.output_variables,
            "output_assembly": OutputAssembly.from_casadi(
                self.output_variables,
                self.computed_var_fcns,
                time_integrals=self._time_integral_vars,
            ),
            "var_fcns": self.computed_var_fcns,
            "var_idaklu_fcns": [],
            "dvar_dy_idaklu_fcns": [],
            "dvar_dp_idaklu_fcns": [],
        }

        for name in _SETUP_FCN_KEYS:
            self._setup[name] = to_idaklu(fns[name])

        # The idaklu-bound Function isn't callable from Python; keep the
        # original for the closest_event_idx lookup in _post_process_solution.
        self._setup["rootfn_casadi"] = fns["rootfn"]

        for key in self.output_variables:
            self._setup["var_idaklu_fcns"].append(to_idaklu(fns[f"var:{key}"]))
            if has_sens:
                self._setup["dvar_dy_idaklu_fcns"].append(
                    to_idaklu(fns[f"dvar_dy:{key}"])
                )
                self._setup["dvar_dp_idaklu_fcns"].append(
                    to_idaklu(fns[f"dvar_dp:{key}"])
                )

        self._setup["solver"] = idaklu.create_casadi_solver_group(
            number_of_states=self._setup["number_of_states"],
            number_of_parameters=self._setup["number_of_sensitivity_parameters"],
            rhs_alg=self._setup["rhs_algebraic"],
            jac_times_cjmass=self._setup["jac_times_cjmass"],
            jac_times_cjmass_colptrs=self._setup["jac_times_cjmass_colptrs"],
            jac_times_cjmass_rowvals=self._setup["jac_times_cjmass_rowvals"],
            jac_times_cjmass_nnz=self._setup["jac_times_cjmass_nnz"],
            jac_bandwidth_lower=self._setup["jac_bandwidth_lower"],
            jac_bandwidth_upper=self._setup["jac_bandwidth_upper"],
            jac_action=self._setup["jac_rhs_algebraic_action"],
            mass_action=self._setup["mass_action"],
            sens=self._setup["sensfn"],
            events=self._setup["rootfn"],
            number_of_events=self._setup["num_of_events"],
            rhs_alg_id=self._setup["ids"],
            atol=self._setup["atol"],
            rtol=self.rtol,
            inputs=self._setup["inputs"],
            var_fcns=self._setup["var_idaklu_fcns"],
            dvar_dy_fcns=self._setup["dvar_dy_idaklu_fcns"],
            dvar_dp_fcns=self._setup["dvar_dp_idaklu_fcns"],
            options=self._options,
            alg_res=self._setup["alg_res"],
            alg_jac=self._setup["alg_jac"],
        )

        # C++ solver group now owns each casadi::Function via shared_ptr;
        # drop the Python references. rootfn_casadi stays for runtime use.
        for key in (*_SETUP_FCN_KEYS, *_SETUP_FCN_LIST_KEYS):
            self._setup.pop(key, None)

        # Release the public casadi.Function caches now that the C++ group
        # owns the functions. _setup["var_fcns"] keeps the references that
        # _post_process_solution still needs; the dvar caches are unused
        # after setup, so dropping them frees that memory immediately.
        self.computed_var_fcns = {}
        self.computed_dvar_dy_fcns = {}
        self.computed_dvar_dp_fcns = {}

        return base_set_up_return

    def __getstate__(self):
        # Drop the rebuildable state (C++ solver + casadi.Function graphs)
        # so the next solve() rebuilds from the model rather than shipping
        # serialised functions in the pickle.
        state = self.__dict__.copy()
        for key in _REBUILDABLE_STATE_KEYS:
            state.pop(key, None)
        return state

    def __setstate__(self, d):
        self.__dict__.update(d)
        # Restore the empty defaults BaseSolver.__init__ would set, so the
        # solver reads as "not yet set up" until the next solve().
        self._model_set_up = {}
        self.computed_var_fcns = {}

    @property
    def options(self):
        return self._options

    def _integrate(
        self,
        model,
        t_eval,
        inputs_list: list[dict] | None = None,
        t_interp=None,
        nproc=None,
    ):
        """
        Overloads the _integrate method from BaseSolver to use the IDAKLU solver
        """
        if model.convert_to_format not in ("casadi", "rust"):  # pragma: no cover
            raise pybamm.SolverError("Unsupported IDAKLU solver configuration.")

        inputs_list = inputs_list or [{}]

        # stack inputs so that they are a 2D array of shape (number_of_inputs, number_of_parameters)
        if inputs_list and inputs_list[0]:
            inputs = np.vstack([flatten_inputs(d) for d in inputs_list])
        else:
            inputs = np.array([[]] * len(inputs_list))

        sensitivity_names = self._setup["sensitivity_names"]
        if sensitivity_names and inputs_list and inputs_list[0]:
            pbar = np.vstack(
                [_sensitivity_scales(d, sensitivity_names) for d in inputs_list]
            )
        else:
            pbar = np.empty((0, 0))

        # y0full is now a list with length = number of input sets
        y0full = np.vstack(model.y0full)
        ydot0full = np.vstack(model.ydot0full)

        atol = getattr(model, "atol", self.atol)
        atol = self._check_atol_type(atol, model)

        logger = (
            pybamm.logger.debug if pybamm.logger.isEnabledFor(logging.DEBUG) else None
        )

        timer = pybamm.Timer()
        try:
            solns = self._setup["solver"].solve(
                t_eval,
                t_interp,
                y0full,
                ydot0full,
                inputs,
                pbar,
                logger=logger,
            )
        except ValueError as e:
            # Return from None to replace the C++ runtime error
            raise pybamm.SolverError(str(e)) from None
        integration_time = timer.time()

        return [
            self._post_process_solution(
                soln, model, integration_time, inputs_dict, t_eval
            )
            for soln, inputs_dict in zip(solns, inputs_list, strict=False)
        ]

    @property
    def _internal_initialisation(self) -> bool:
        return bool(self._options["calc_ic"])

    def _check_event_violation_on_initialisation(self, *args, **kwargs):
        if self._internal_initialisation:
            return
        return self._check_event_violation(*args, **kwargs)

    def _check_event_violation_post_solve(self, *args, **kwargs):
        if not self._internal_initialisation:
            return
        return self._check_event_violation(*args, **kwargs)

    def _post_process_solution(self, sol, model, integration_time, inputs_dict, t_eval):
        number_of_sensitivity_parameters = self._setup[
            "number_of_sensitivity_parameters"
        ]
        sensitivity_names = self._setup["sensitivity_names"]
        number_of_timesteps = sol.t.size
        number_of_states = model.len_rhs_and_alg
        save_outputs_only = self.output_variables
        if save_outputs_only:
            # Substitute empty vectors for state vector 'y'
            y_out = np.zeros((number_of_timesteps * number_of_states, 0))
            y_event = sol.y_term
        else:
            y_out = sol.y.reshape((number_of_timesteps, number_of_states))
            y_event = y_out[-1]

        # If there is only one step and an event was found, the event was
        # triggered at t0 after consistent initialization. y_event is the
        # post-IC state: sol.y_term (outputs-only) or y_out[-1] (full),
        # both stored after IC. This check identifies *which* event fired.
        if number_of_timesteps == 1 and sol.flag == _IDA_ROOT_RETURN:
            self._check_event_violation_post_solve(t_eval, model, y_event, inputs_dict)

        # return sensitivity solution, we need to flatten yS to
        # (#timesteps * #states (where t is changing the quickest),)
        # to match format used by Solution
        # note that yS is (n_p, n_t, n_y)
        if number_of_sensitivity_parameters != 0:
            yS_out = {
                name: sol.yS[i].reshape(-1, 1)
                for i, name in enumerate(sensitivity_names)
            }
            # add "all" stacked sensitivities ((#timesteps * #states,#sens_params))
            yS_out["all"] = np.hstack([yS_out[name] for name in sensitivity_names])
        else:
            yS_out = {}

        # IDA_SUCCESS (0) = solved for all t_eval
        # IDA_ROOT_RETURN (2) = found root(s)
        # < 0 = solver failure
        if sol.flag == _IDA_ROOT_RETURN:
            termination = "event"
        elif sol.flag >= 0:
            termination = "final time"
        elif sol.flag < 0:
            termination = "failure"
            msg = idaklu.sundials_error_message(sol.flag)
            match self._on_failure:
                case "warn":
                    warnings.warn(
                        msg + ", returning a partial solution.",
                        stacklevel=2,
                    )
                case "error":
                    raise pybamm.SolverError(msg)

        if sol.yp.size > 0:
            yp = sol.yp.reshape((number_of_timesteps, number_of_states)).T
        else:
            yp = None

        t = sol.t
        t_eval = np.array(t_eval)
        if t[-1] != t_eval[-1]:
            idx_final = np.searchsorted(t_eval, t[-1]) + 1
            t_eval = t_eval[:idx_final]

            # the final index may differ due to an event;
            # manually set it to the true final time
            t_eval[-1] = t[-1]

        # Forward the compile flag so post-solve observation uses the same
        # backend as the integration.
        solution_options = {"compile": self._options["compile"]}

        newsol = pybamm.Solution(
            t,
            np.transpose(y_out),
            model,
            inputs_dict,
            np.array([sol.t[-1]]),
            np.transpose(y_event)[:, np.newaxis],
            termination,
            all_sensitivities=yS_out,
            all_yps=yp,
            all_t_evals=t_eval,
            variables_returned=bool(save_outputs_only),
            options=solution_options,
        )

        # Fast path via the compiled events (`rootfn_casadi`, or the rust model's
        # own event tapes); else BaseSolver re-walks events symbolically.
        if sol.flag == _IDA_ROOT_RETURN and self._setup.get("num_of_events", 0) > 0:
            t_event = float(sol.t[-1])
            y_event_flat = np.asarray(y_event).reshape(-1)
            if "rootfn_casadi" in self._setup:
                event_values = np.asarray(
                    self._setup["rootfn_casadi"](
                        t_event,
                        y_event_flat,
                        flatten_inputs(inputs_dict),
                    )
                ).reshape(-1)
                newsol.closest_event_idx = int(np.nanargmin(np.abs(event_values)))
            elif model.convert_to_format == "rust":
                stacked = stack_inputs(inputs_dict, "rust")
                event_values = np.array(
                    [
                        float(np.asarray(fn(t_event, y_event_flat, stacked)).ravel()[0])
                        for fn in model.terminate_events_eval
                    ]
                )
                newsol.closest_event_idx = int(np.nanargmin(np.abs(event_values)))

        newsol.integration_time = integration_time
        newsol.solver_statistics = pybamm.SolverStatistics(
            number_of_steps=sol.stats.number_of_steps,
            number_of_linear_solver_setups=sol.stats.number_of_linear_solver_setups,
            number_of_nonlinear_solver_iterations=sol.stats.number_of_nonlinear_solver_iterations,
            number_of_nonlinear_solver_fails=sol.stats.number_of_nonlinear_solver_fails,
            number_of_error_test_failures=sol.stats.number_of_error_test_failures,
        )
        if self._observes_via_compiled_model and model.convert_to_format == "rust":
            # Attached on the outputs-only path too: first_state/last_state carry
            # full states, and their observation must match the full-state path.
            # IDAKLU always stores yps, so every solution reads off-grid.
            newsol.observation = NativeInterpolatingObservation.uniform(
                self._setup["rust_model"],
                len(newsol.all_ys),
                cache=self._setup["rust_observation_cache"],
            )
        if not save_outputs_only:
            return newsol

        # On this path `sol.y` carries the concatenated outputs rather than the
        # states, and `sol.yS` their sensitivities as (n_t, n_rows, n_p).
        self._setup["output_assembly"].attach(
            newsol,
            np.asarray(sol.y).reshape(number_of_timesteps, -1),
            sensitivities=(
                np.asarray(sol.yS) if number_of_sensitivity_parameters else None
            ),
            sensitivity_names=sensitivity_names,
        )
        return newsol

    def _set_consistent_initialization(self, model, time, inputs_list):
        """
        Initialize y0 and ydot0 for the solver. In addition to calculating
        y0 from BaseSolver, we also calculate ydot0 for semi-explicit DAEs

        Parameters
        ----------
        model : :class:`pybamm.BaseModel`
            The model for which to calculate initial conditions.
        time : numeric type
            The time at which to calculate the initial conditions.
        inputs_list : list of dict
            Any input parameters to pass to the model when solving.
        """

        # set model.y0_list
        super()._set_consistent_initialization(model, time, inputs_list)

        def handle_y0(y0):
            if isinstance(y0, casadi.DM):
                y0 = y0.full()
            return y0.flatten()

        y0_list = [handle_y0(y0) for y0 in model.y0_list]

        # calculate the time derivatives of the differential equations
        # for semi-explicit DAEs
        if model.len_rhs > 0:
            ydot0_list = [
                self._rhs_dot_consistent_initialization(y0, model, time, inputs_dict)
                for y0, inputs_dict in zip(y0_list, inputs_list, strict=True)
            ]
        else:
            ydot0_list = [np.zeros_like(y0) for y0 in y0_list]

        sensitivity = model.y0S_list and model.uses_stacked_inputs
        if sensitivity:
            y0S_list = model.y0S_list
            y0full = []
            ydot0full = []
            for y0, ydot0, y0S, inputs_dict in zip(
                y0_list, ydot0_list, y0S_list, inputs_list, strict=True
            ):
                y0f, ydot0f = self._sensitivity_consistent_initialization(
                    y0, ydot0, y0S, time, inputs_dict
                )
                y0full.append(y0f)
                ydot0full.append(ydot0f)
        else:
            y0full = y0_list
            ydot0full = ydot0_list

        model.y0full = y0full
        model.ydot0full = ydot0full

    def _rhs_dot_consistent_initialization(self, y0, model, time, inputs_dict):
        """
        Compute the consistent initialization of ydot0 for the differential terms
        for the solver. If we have a semi-explicit DAE, we can explicitly solve
        for this value using the consistently initialized y0 vector.

        Parameters
        ----------
        y0 : :class:`numpy.array`
            The initial values of the state vector.
        model : :class:`pybamm.BaseModel`
            The model for which to calculate initial conditions.
        time : numeric type
            The time at which to calculate the initial conditions.
        inputs_dict : dict
            Any input parameters to pass to the model when solving.

        """
        inputs_dict = inputs_dict or {}
        if model.uses_stacked_inputs:
            input_eval = stack_inputs(inputs_dict, model.convert_to_format)
        else:
            input_eval = inputs_dict

        ydot0 = np.zeros_like(y0)
        rhs0 = model.rhs_eval(time, y0, input_eval)
        if isinstance(rhs0, casadi.DM):
            rhs0 = rhs0.full()
        rhs0 = rhs0.flatten()

        # for the differential terms, ydot = M^-1 * (rhs)
        if model.is_standard_form_dae:
            # M^-1 is the identity matrix, so we can just use rhs
            ydot0[: model.len_rhs] = rhs0
        else:
            # M^-1 is not the identity matrix, so we need to use the mass matrix
            M_ode = model.mass_matrix.entries[: model.len_rhs, : model.len_rhs]
            ydot0[: model.len_rhs] = spsolve(M_ode, rhs0)

        return ydot0

    def _sensitivity_consistent_initialization(self, y0, ydot0, y0S, time, inputs_dict):
        """
        Extend the consistent initialization to include the sensitivty equations

        Parameters
        ----------
        y0 : :class:`numpy.array`
            The initial values of the state vector.
        ydot0 : :class:`numpy.array`
            The initial values of the time derivatives of the state vector.
        y0S : :class:`numpy.array`
            The initial values of the sensitivity state vectors.
        time : numeric type
            The time at which to calculate the initial conditions.
        inputs_dict : dict
            Any input parameters to pass to the model when solving.

        """

        if isinstance(y0S, casadi.DM | np.ndarray):
            y0S = (y0S,)

        if y0S and isinstance(y0S[0], casadi.DM):
            y0S = (x.full() for x in y0S)
        y0S = [x.flatten() for x in y0S]

        y0full = np.concatenate([y0, *y0S])

        ydot0S = [np.zeros_like(y0S_i) for y0S_i in y0S]
        ydot0full = np.concatenate([ydot0, *ydot0S])

        return y0full, ydot0full

    def jaxify(
        self,
        model,
        t_eval,
        *,
        output_variables=None,
        calculate_sensitivities=True,
        t_interp=None,
    ):
        """JAXify the solver object

        Creates a JAX expression representing the IDAKLU-wrapped solver
        object.

        Parameters
        ----------
        model : :class:`pybamm.BaseModel`
            The model to be solved
        t_eval : numeric type, optional
            The times at which to stop the integration due to a discontinuity in time.
        output_variables : list of str, optional
            The variables to be returned. If None, all variables in the model are used.
        calculate_sensitivities : bool, optional
            Whether to calculate sensitivities. Default is True.
        t_interp : None, list or ndarray, optional
            The times (in seconds) at which to interpolate the solution. Defaults to `None`,
            which returns the adaptive time-stepping times.
        """
        obj = pybamm.IDAKLUJax(
            self,  # IDAKLU solver instance
            model,
            t_eval,
            output_variables=output_variables,
            calculate_sensitivities=calculate_sensitivities,
            t_interp=t_interp,
        )
        return obj

    def reduce_solution(
        self,
        solution,
        hermite_reduction_factor=2.0,
    ) -> pybamm.Solution:
        """Reduce knots in a pybamm.Solution using Hermite spline compression.

        The multiplier M controls the total error budget relative to the
        solver's own WRMS tolerance.  The knot reducer is allowed an
        additional WRMS error of (M-1), so the total error satisfies
        ``||e_total||_WRMS <= M``.  M = 2 (default) means the reduced
        solution may have up to 2x the solver's local error.

        Parameters
        ----------
        solution : :class:`pybamm.Solution`
            The solution to reduce. Must have hermite interpolation data
            (all_yps is not None).
        hermite_reduction_factor : float, optional
            Total error multiplier (>= 1.0, default 2.0). The knot
            reducer's WRMS threshold is N * (M-1)^2. Larger values
            allow more aggressive reduction.

        Returns
        -------
        :class:`pybamm.Solution`
            The modified solution.
        """
        if not solution.hermite_interpolation:
            raise pybamm.SolverError(
                "reduce_solution requires Hermite interpolation data (all_yps)."
            )
        if self.options["hermite_reduction_factor"] != 1.0:
            raise pybamm.SolverError(
                "reduce_solution requires the original solver to have "
                "`hermite_reduction_factor = 1.0`"
            )

        all_ts = solution.all_ts
        all_ys = solution.all_ys
        all_yps = solution.all_yps
        all_models = solution.all_models
        n_seg = len(all_ts)

        rtol = self.rtol
        atol = self.atol

        # Build flat time-major arrays for the C++ reducer.
        # all_ys[i] is (n_states, M) transposed view whose underlying
        # buffer is already time-major (M * n_states,).  .T.ravel()
        # returns a 1D view of that buffer -- no copy.
        flat_ys = [all_ys[i].T.ravel() for i in range(n_seg)]
        flat_yps = [all_yps[i].T.ravel() for i in range(n_seg)]

        # Build per-segment atol vectors
        atol_vecs = [None] * n_seg
        for i, model in enumerate(all_models):
            atol_vecs[i] = self._check_atol_type(atol, model)

        ts_vec = idaklu.VectorRealtypeNdArray(all_ts)
        ys_vec = idaklu.VectorRealtypeNdArray(flat_ys)
        yps_vec = idaklu.VectorRealtypeNdArray(flat_yps)
        atols_vec = idaklu.VectorRealtypeNdArray(atol_vecs)
        t_evals_vec = idaklu.VectorRealtypeNdArray(solution.all_t_evals)

        red_ts, red_ys, red_yps = idaklu.reduce_knots(
            ts_vec,
            ys_vec,
            yps_vec,
            atols_vec,
            t_evals_vec,
            float(rtol),
            float(hermite_reduction_factor),
        )

        # Reshape reduced flat arrays back to (n_states, K) convention
        new_ts = [np.asarray(red_ts[i]) for i in range(n_seg)]
        new_ys = []
        new_yps = []
        for i in range(n_seg):
            K = len(new_ts[i])
            N = all_ys[i].shape[0]
            new_ys.append(np.asarray(red_ys[i]).reshape(K, N).T)
            new_yps.append(np.asarray(red_yps[i]).reshape(K, N).T)

        new_sol = pybamm.Solution(
            all_ts=new_ts,
            all_ys=new_ys,
            all_yps=new_yps,
            all_models=all_models,
            all_inputs=solution.all_inputs,
            t_event=solution.t_event,
            y_event=solution.y_event,
            termination=solution.termination,
            all_sensitivities=solution._all_sensitivities,
            all_t_evals=solution.all_t_evals,
            variables_returned=solution.variables_returned,
            options=solution.user_options,
        )

        # Propagate metadata from the original solution. Thinning knots leaves
        # the segments and their models alone, so the backend carries over as is.
        new_sol.observation = solution.observation
        new_sol._all_inputs_stacked = solution.all_inputs_stacked
        new_sol._all_inputs_casadi = solution.all_inputs_casadi
        new_sol.closest_event_idx = solution.closest_event_idx

        new_sol.solve_time = solution.solve_time
        new_sol.integration_time = solution.integration_time
        new_sol.set_up_time = solution.set_up_time

        return new_sol
