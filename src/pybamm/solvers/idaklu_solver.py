# mypy: ignore-errors
import math
import numbers
import warnings

import casadi
import numpy as np
import pybammsolvers.idaklu as idaklu

import pybamm


class IDAKLUSolver(pybamm.BaseSolver):
    """
    Solve a discretised model, using sundials with the KLU sparse linear solver.

    Parameters
    ----------
    rtol : float, optional
        The relative tolerance for the solver (default is 1e-4).
    atol : float, optional
        The absolute tolerance for the solver (default is 1e-6).
    root_method : str or pybamm algebraic solver class, optional
        The method to use to find initial conditions (for DAE solvers).
        If a solver class, must be an algebraic solver class.
        If "casadi",
        the solver uses casadi's Newton rootfinding algorithm to find initial
        conditions. Otherwise, the solver uses 'scipy.optimize.root' with method
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
        Default is "raise".
    output_variables : list[str], optional
        List of variables to calculate and return. If none are specified then
        the complete state vector is returned (can be very large) (default is [])
    options: dict, optional
        Addititional options to pass to the solver, by default:

        .. code-block:: python

            options = {
                # Print statistics of the solver after every solve
                "print_stats": False,
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
                "max_linesearch_backtracks_ic": 100,
                # Turn off linesearch
                "linesearch_off_ic": False,
                # How to calculate the initial conditions.
                # "True": calculate all y0 given ydot0
                # "False": calculate y_alg0 and ydot_diff0 given y_diff0
                "init_all_y_ic": False,
                # Calculate consistent initial conditions
                "calc_ic": True,
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

        Note: These options only have an effect if model.convert_to_format == 'casadi'


    """

    def __init__(
        self,
        rtol=1e-4,
        atol=1e-6,
        root_method="casadi",
        root_tol=1e-6,
        extrap_tol=None,
        on_extrapolation=None,
        output_variables=None,
        on_failure=None,
        options=None,
    ):
        # set default options,
        # (only if user does not supply)
        default_options = {
            "print_stats": False,
            "jacobian": "sparse",
            "preconditioner": "BBDP",
            "precon_half_bandwidth": 5,
            "precon_half_bandwidth_keep": 5,
            "num_threads": 1,
            "num_solvers": 1,
            "linear_solver": "SUNLinSol_KLU",
            "linsol_max_iterations": 5,
            "epsilon_linear_tolerance": 0.05,
            "increment_factor": 1.0,
            "linear_solution_scaling": True,
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
            "nonlinear_convergence_coefficient_ic": 0.0033,
            "max_num_steps_ic": 50,
            "max_num_jacobians_ic": 40,
            "max_num_iterations_ic": 100,
            "max_linesearch_backtracks_ic": 100,
            "linesearch_off_ic": False,
            "init_all_y_ic": False,
            "calc_ic": True,
            "num_steps_no_progress": 0,
            "t_no_progress": 0.0,
        }
        if options is None:
            options = default_options
        else:
            if "num_threads" in options and "num_solvers" not in options:
                options["num_solvers"] = options["num_threads"]
            for key, value in default_options.items():
                if key not in options:
                    options[key] = value
        self._options = options

        self.output_variables = [] if output_variables is None else output_variables

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
        )
        self.name = "IDA KLU solver"
        self._supports_interp = True
        self._supports_t_eval_discontinuities = True

        pybamm.citations.register("Hindmarsh2000")
        pybamm.citations.register("Hindmarsh2005")

    def _check_atol_type(self, atol, size):
        """
        This method checks that the atol vector is of the right shape and
        type.

        Parameters
        ----------
        atol: double or np.array or list
            Absolute tolerances. If this is a vector then each entry corresponds to
            the absolute tolerance of one entry in the state vector.
        size: int
            The length of the atol vector
        """

        if isinstance(atol, float):
            atol = atol * np.ones(size)
        elif not isinstance(atol, np.ndarray):
            raise pybamm.SolverError(
                "Absolute tolerances must be a numpy array or float"
            )

        return atol

    def set_up(self, model, inputs=None, t_eval=None, ics_only=False):
        base_set_up_return = super().set_up(model, inputs, t_eval, ics_only)

        inputs_dict = inputs or {}
        # stack inputs
        if inputs_dict:
            arrays_to_stack = [np.array(x).reshape(-1, 1) for x in inputs_dict.values()]
            inputs = np.vstack(arrays_to_stack)
        else:
            inputs = np.array([[]])

        y0 = model.y0
        if isinstance(y0, casadi.DM):
            y0 = y0.full()
        y0 = y0.flatten()

        if ics_only:
            return base_set_up_return

        if model.convert_to_format != "casadi":
            msg = "The python-idaklu solver has been deprecated."
            warnings.warn(msg, DeprecationWarning, stacklevel=2)
            raise pybamm.SolverError(
                f"Unsupported option for convert_to_format={model.convert_to_format}"
            )

        if self._options["jacobian"] == "dense":
            mass_matrix = casadi.DM(model.mass_matrix.entries.toarray())
        else:
            mass_matrix = casadi.DM(model.mass_matrix.entries)

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

        jac_times_cjmass = casadi.Function(
            "jac_times_cjmass",
            [t_casadi, y_casadi, p_casadi_stacked, cj_casadi],
            [
                model.jac_rhs_algebraic_eval(t_casadi, y_casadi, p_casadi_stacked)
                - cj_casadi * mass_matrix
            ],
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

        # also need the action of the mass matrix on a vector
        mass_action = casadi.Function(
            "mass_action", [v_casadi], [casadi.densify(mass_matrix @ v_casadi)]
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
        rhs_ids = np.ones(model.rhs_eval(0, y0, inputs).shape[0])
        alg_ids = np.zeros(len(y0) - len(rhs_ids))
        ids = np.concatenate((rhs_ids, alg_ids))

        number_of_sensitivity_parameters = 0
        if model.jacp_rhs_algebraic_eval is not None:
            sensitivity_names = model.calculate_sensitivities
            if model.convert_to_format == "casadi":
                number_of_sensitivity_parameters = model.jacp_rhs_algebraic_eval.n_out()
            else:
                number_of_sensitivity_parameters = len(sensitivity_names)
        else:
            sensitivity_names = []

        # for the casadi solver we just give it dFdp_i
        if model.jacp_rhs_algebraic_eval is None:
            sensfn = casadi.Function("sensfn", [], [])
        else:
            sensfn = model.jacp_rhs_algebraic_eval

        atol = getattr(model, "atol", self.atol)
        atol = self._check_atol_type(atol, y0.size)

        # Serialize casadi functions
        idaklu_solver_fcn = idaklu.create_casadi_solver_group
        rhs_algebraic_pkl = rhs_algebraic.serialize()
        rhs_algebraic = idaklu.generate_function(rhs_algebraic_pkl)
        jac_times_cjmass_pkl = jac_times_cjmass.serialize()
        jac_times_cjmass = idaklu.generate_function(jac_times_cjmass_pkl)
        jac_rhs_algebraic_action_pkl = jac_rhs_algebraic_action.serialize()
        jac_rhs_algebraic_action = idaklu.generate_function(
            jac_rhs_algebraic_action_pkl
        )
        rootfn_pkl = rootfn.serialize()
        rootfn = idaklu.generate_function(rootfn_pkl)
        mass_action_pkl = mass_action.serialize()
        mass_action = idaklu.generate_function(mass_action_pkl)
        sensfn_pkl = sensfn.serialize()
        sensfn = idaklu.generate_function(sensfn_pkl)

        # if output_variables specified then convert 'variable' casadi
        # function expressions to idaklu-compatible functions
        self.var_idaklu_fcns = []
        self.var_idaklu_fcns_pkl = []
        self.dvar_dy_idaklu_fcns = []
        self.dvar_dy_idaklu_fcns_pkl = []
        self.dvar_dp_idaklu_fcns = []
        self.dvar_dp_idaklu_fcns_pkl = []
        for key in self.output_variables:
            self.var_idaklu_fcns_pkl.append(self.computed_var_fcns[key].serialize())
            self.var_idaklu_fcns.append(
                idaklu.generate_function(self.var_idaklu_fcns_pkl[-1])
            )
            # Convert derivative functions for sensitivities
            if (len(inputs) > 0) and (model.calculate_sensitivities):
                self.dvar_dy_idaklu_fcns_pkl.append(
                    self.computed_dvar_dy_fcns[key].serialize()
                )
                self.dvar_dy_idaklu_fcns.append(
                    idaklu.generate_function(self.dvar_dy_idaklu_fcns_pkl[-1])
                )
                self.dvar_dp_idaklu_fcns_pkl.append(
                    self.computed_dvar_dp_fcns[key].serialize()
                )
                self.dvar_dp_idaklu_fcns.append(
                    idaklu.generate_function(self.dvar_dp_idaklu_fcns_pkl[-1])
                )

        self._setup = {
            "number_of_states": len(y0),
            "inputs": len(inputs),
            "solver_function": idaklu_solver_fcn,  # callable
            "jac_bandwidth_upper": jac_bw_upper,  # int
            "jac_bandwidth_lower": jac_bw_lower,  # int
            "atol": atol,
            "rhs_algebraic": rhs_algebraic,  # function
            "rhs_algebraic_pkl": rhs_algebraic_pkl,
            "jac_times_cjmass": jac_times_cjmass,  # function
            "jac_times_cjmass_pkl": jac_times_cjmass_pkl,
            "jac_times_cjmass_colptrs": jac_times_cjmass_colptrs,  # array
            "jac_times_cjmass_rowvals": jac_times_cjmass_rowvals,  # array
            "jac_times_cjmass_nnz": jac_times_cjmass_nnz,  # int
            "jac_rhs_algebraic_action": jac_rhs_algebraic_action,  # function
            "jac_rhs_algebraic_action_pkl": jac_rhs_algebraic_action_pkl,
            "mass_action": mass_action,  # function
            "mass_action_pkl": mass_action_pkl,
            "sensfn": sensfn,  # function
            "sensfn_pkl": sensfn_pkl,
            "rootfn": rootfn,  # function
            "rootfn_pkl": rootfn_pkl,
            "num_of_events": num_of_events,  # int
            "ids": ids,  # array
            "sensitivity_names": sensitivity_names,
            "number_of_sensitivity_parameters": number_of_sensitivity_parameters,
            "standard_form_dae": model.is_standard_form_dae,  # bool
            "output_variables": self.output_variables,
            "var_fcns": self.computed_var_fcns,
            "var_idaklu_fcns": self.var_idaklu_fcns,
            "var_idaklu_fcns_pkl": self.var_idaklu_fcns_pkl,
            "dvar_dy_idaklu_fcns": self.dvar_dy_idaklu_fcns,
            "dvar_dy_idaklu_fcns_pkl": self.dvar_dy_idaklu_fcns_pkl,
            "dvar_dp_idaklu_fcns": self.dvar_dp_idaklu_fcns,
            "dvar_dp_idaklu_fcns_pkl": self.dvar_dp_idaklu_fcns_pkl,
        }

        solver = self._setup["solver_function"](
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
        )

        self._setup["solver"] = solver

        return base_set_up_return

    def __getstate__(self):
        # if _setup is not defined then we haven't called set_up yet
        if not hasattr(self, "_setup"):
            return self.__dict__

        self.var_idaklu_fcns = []

        for key in [
            "solver",
            "solver_function",
            "rhs_algebraic",
            "jac_times_cjmass",
            "jac_rhs_algebraic_action",
            "mass_action",
            "sensfn",
            "rootfn",
            "var_idaklu_fcns",
            "dvar_dy_idaklu_fcns",
            "dvar_dp_idaklu_fcns",
        ]:
            self._setup.pop(key, None)
        return self.__dict__

    def __setstate__(self, d):
        self.__dict__.update(d)

        # if _setup is not defined then we haven't called set_up yet
        if not hasattr(self, "_setup"):
            return

        self.var_idaklu_fcns = [
            idaklu.generate_function(f) for f in self.var_idaklu_fcns_pkl
        ]

        for key in [
            "rhs_algebraic",
            "jac_times_cjmass",
            "jac_rhs_algebraic_action",
            "mass_action",
            "sensfn",
            "rootfn",
        ]:
            self._setup[key] = idaklu.generate_function(self._setup[key + "_pkl"])

        for key in [
            "var_idaklu_fcns",
            "dvar_dy_idaklu_fcns",
            "dvar_dp_idaklu_fcns",
        ]:
            self._setup[key] = [
                idaklu.generate_function(f) for f in self._setup[key + "_pkl"]
            ]

        self._setup["solver_function"] = idaklu.create_casadi_solver_group

        self._setup["solver"] = self._setup["solver_function"](
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
        )

    @property
    def supports_parallel_solve(self):
        return True

    @property
    def options(self):
        return self._options

    def _apply_solver_initial_conditions(self, model, initial_conditions):
        """
        Apply custom initial conditions to a model by overriding model.y0.

        Parameters
        ----------
        model : pybamm.BaseModel
            A model with a precomputed y0 vector.
        initial_conditions : dict or numpy.ndarray
            Either a mapping from variable names to values (scalar or array),
            or a flat numpy array matching the length of model.y0.
        """
        if isinstance(initial_conditions, dict):
            y0_np = (
                model.y0.full() if isinstance(model.y0, casadi.DM) else model.y0.copy()
            )

            for var_name, value in initial_conditions.items():
                found = False
                for symbol, slice_info in model.y_slices.items():
                    if symbol.name == var_name:
                        var_slice = slice_info[0]
                        y0_np[var_slice] = value
                        found = True
                        break
                if not found:
                    raise ValueError(f"Variable '{var_name}' not found in model")

            model.y0 = casadi.DM(y0_np)

        elif isinstance(initial_conditions, np.ndarray):
            model.y0 = casadi.DM(initial_conditions)
        else:
            raise TypeError("Initial conditions must be dict or numpy array")

    def _integrate(
        self, model, t_eval, inputs_list=None, t_interp=None, initial_conditions=None
    ):
        """
        Solve a DAE model defined by residuals with initial conditions y0.

        Parameters
        ----------
        model : :class:`pybamm.BaseModel`
            The model whose solution to calculate.
        t_eval : numeric type
            The times at which to stop the integration due to a discontinuity in time.
        inputs_list: list of dict, optional
            Any input parameters to pass to the model when solving.
        t_interp : None, list or ndarray, optional
            The times (in seconds) at which to interpolate the solution. Defaults to `None`,
            which returns the adaptive time-stepping times.
        initial_conditions : dict, numpy.ndarray, or list, optional
            Override the model’s default `y0`.  Can be:

            - a dict mapping variable names → values
            - a 1D array of length `n_states`
            - a list of such overrides (one per parallel solve)

        """
        if model.convert_to_format != "casadi":  # pragma: no cover
            # Shouldn't ever reach this point
            raise pybamm.SolverError("Unsupported IDAKLU solver configuration.")

        inputs_list = inputs_list or [{}]

        # stack inputs so that they are a 2D array of shape (number_of_inputs, number_of_parameters)
        if inputs_list and inputs_list[0]:
            inputs = np.vstack(
                [
                    np.hstack([np.array(x).reshape(-1) for x in inputs_dict.values()])
                    for inputs_dict in inputs_list
                ]
            )
        else:
            inputs = np.array([[]] * len(inputs_list))

        if initial_conditions is not None:
            if isinstance(initial_conditions, list):
                if len(initial_conditions) != len(inputs_list):
                    raise ValueError(
                        "Number of initial conditions must match number of input sets"
                    )

                y0_list = []

                model_copy = model.new_copy()
                for ic in initial_conditions:
                    self._apply_solver_initial_conditions(model_copy, ic)
                    y0_list.append(model_copy.y0.full().flatten())

                y0full = np.vstack(y0_list)
                ydot0full = np.zeros_like(y0full)

            else:
                self._apply_solver_initial_conditions(model, initial_conditions)

                y0_np = model.y0.full()

                y0full = np.vstack([y0_np for _ in range(len(inputs_list))])
                ydot0full = np.zeros_like(y0full)
        else:
            # stack y0full and ydot0full so they are a 2D array of shape (number_of_inputs, number_of_states + number_of_parameters * number_of_states)
            # note that y0full and ydot0full are currently 1D arrays (i.e. independent of inputs), but in the future we will support
            # different initial conditions for different inputs. For now we just repeat the same initial conditions for each input
            y0full = np.vstack([model.y0full] * len(inputs_list))
            ydot0full = np.vstack([model.ydot0full] * len(inputs_list))

        atol = getattr(model, "atol", self.atol)
        atol = self._check_atol_type(atol, y0full.size)

        timer = pybamm.Timer()
        solns = self._setup["solver"].solve(
            t_eval,
            t_interp,
            y0full,
            ydot0full,
            inputs,
        )
        integration_time = timer.time()

        return [
            self._post_process_solution(soln, model, integration_time, inputs_dict)
            for soln, inputs_dict in zip(solns, inputs_list, strict=False)
        ]

    def _post_process_solution(self, sol, model, integration_time, inputs_dict):
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

        # 0 = solved for all t_eval
        # 2 = found root(s)
        # < 0 = solver failure
        if sol.flag == 2:
            termination = "event"
        elif sol.flag >= 0:
            termination = "final time"
        elif sol.flag < 0:
            termination = "failure"
            match self._on_failure:
                case "warn":
                    warnings.warn(
                        f"FAILURE {self._solver_flag(sol.flag)}, returning a partial solution.",
                        stacklevel=2,
                    )
                case "raise":
                    raise pybamm.SolverError(f"FAILURE {self._solver_flag(sol.flag)}")

        if sol.yp.size > 0:
            yp = sol.yp.reshape((number_of_timesteps, number_of_states)).T
        else:
            yp = None

        newsol = pybamm.Solution(
            sol.t,
            np.transpose(y_out),
            model,
            inputs_dict,
            np.array([sol.t[-1]]),
            np.transpose(y_event)[:, np.newaxis],
            termination,
            all_sensitivities=yS_out,
            all_yps=yp,
            variables_returned=bool(save_outputs_only),
        )

        newsol.integration_time = integration_time
        if not save_outputs_only:
            return newsol

        # Populate variables and sensitivities dictionaries directly
        number_of_samples = sol.y.shape[0] // number_of_timesteps
        sol.y = sol.y.reshape((number_of_timesteps, number_of_samples))
        sensitivity_params = (
            list(inputs_dict.keys()) if model.calculate_sensitivities else []
        )

        start_idx = 0
        for var in self.output_variables:
            var_nnz, var_shape, base_variables = self._get_variable_info(model, var)
            end_idx = start_idx + var_nnz
            data = sol.y[:, start_idx:end_idx]
            time_indep = False

            # handle any time integral variables
            if var in self._time_integral_vars:
                # time integral variables should all be 1D
                tiv = self._time_integral_vars[var]
                data = tiv.postfix(data.reshape(-1), sol.t, inputs_dict)
                time_indep = True

            newsol._variables[var] = pybamm.ProcessedVariableComputed(
                [model.variables_and_events[var]],
                base_variables,
                [data],
                newsol,
                time_indep=time_indep,
            )

            # Add sensitivities
            newsol[var]._sensitivities = {}
            if sensitivity_params:
                if var_nnz != math.prod(var_shape):
                    raise pybamm.SolverError(
                        f"Sensitivity of sparse variables not supported. {var} is a sparse variable with number of non-zeros {var_nnz} and shape {var_shape}"
                    )
                sens_data = sol.yS[:, start_idx:end_idx, :]
                sens_data = sens_data.reshape(
                    number_of_timesteps * (end_idx - start_idx),
                    number_of_sensitivity_parameters,
                )
                if var in self._time_integral_vars:
                    tiv = self._time_integral_vars[var]
                    sens_data = tiv.postfix_sensitivities(
                        var, data, sol.t, inputs_dict, sens_data
                    )
                newsol[var]._sensitivities["all"] = sens_data

                # Add the individual sensitivity
                for i, name in enumerate(inputs_dict.keys()):
                    sens = newsol[var]._sensitivities["all"][:, i : i + 1].reshape(-1)
                    newsol[var]._sensitivities[name] = sens

            start_idx += var_nnz
        return newsol

    def _get_variable_info(self, model, var) -> tuple:
        """Get variable length and base variables based on model format."""
        if model.convert_to_format == "casadi":
            base_var = self._setup["var_fcns"][var]
            var_eval = base_var(0.0, 0.0, 0.0)
            var_nnz = var_eval.sparsity().nnz()
            var_shape = var_eval.shape
            return var_nnz, var_shape, [base_var]
        else:  # pragma: no cover
            raise pybamm.SolverError(
                f"Unsupported evaluation engine for convert_to_format="
                f"{model.convert_to_format}"
            )

    def _set_consistent_initialization(self, model, time, inputs_dict):
        """
        Initialize y0 and ydot0 for the solver. In addition to calculating
        y0 from BaseSolver, we also calculate ydot0 for semi-explicit DAEs

        Parameters
        ----------
        model : :class:`pybamm.BaseModel`
            The model for which to calculate initial conditions.
        time : numeric type
            The time at which to calculate the initial conditions.
        inputs_dict : dict
            Any input parameters to pass to the model when solving.
        """

        # set model.y0
        super()._set_consistent_initialization(model, time, inputs_dict)

        casadi_format = model.convert_to_format == "casadi"

        y0 = model.y0
        if isinstance(y0, casadi.DM):
            y0 = y0.full()
        y0 = y0.flatten()

        # calculate the time derivatives of the differential equations
        # for semi-explicit DAEs
        if model.len_rhs > 0:
            ydot0 = self._rhs_dot_consistent_initialization(
                y0, model, time, inputs_dict
            )
        else:
            ydot0 = np.zeros_like(y0)

        sensitivity = (model.y0S is not None) and casadi_format
        if sensitivity:
            y0full, ydot0full = self._sensitivity_consistent_initialization(
                y0, ydot0, model, time, inputs_dict
            )
        else:
            y0full = y0
            ydot0full = ydot0

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
        casadi_format = model.convert_to_format == "casadi"

        inputs_dict = inputs_dict or {}
        # stack inputs
        if inputs_dict:
            arrays_to_stack = [np.array(x).reshape(-1, 1) for x in inputs_dict.values()]
            inputs = np.vstack(arrays_to_stack)
        else:
            inputs = np.array([[]])

        ydot0 = np.zeros_like(y0)
        # calculate the time derivatives of the differential equations
        input_eval = inputs if casadi_format else inputs_dict

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
            ydot0[: model.len_rhs] = model.mass_matrix_inv.entries @ rhs0

        return ydot0

    def _sensitivity_consistent_initialization(
        self, y0, ydot0, model, time, inputs_dict
    ):
        """
        Extend the consistent initialization to include the sensitivty equations

        Parameters
        ----------
        y0 : :class:`numpy.array`
            The initial values of the state vector.
        ydot0 : :class:`numpy.array`
            The initial values of the time derivatives of the state vector.
        time : numeric type
            The time at which to calculate the initial conditions.
        model : :class:`pybamm.BaseModel`
            The model for which to calculate initial conditions.
        inputs_dict : dict
            Any input parameters to pass to the model when solving.

        """

        y0S = model.y0S

        if isinstance(y0S, casadi.DM):
            y0S = (y0S,)

        if isinstance(y0S[0], casadi.DM):
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

    @staticmethod
    def _solver_flag(flag):
        flags = {
            99: "IDA_WARNING: IDASolve succeeded but an unusual situation occurred.",
            2: "IDA_ROOT_RETURN: IDASolve succeeded and found one or more roots.",
            1: "IDA_TSTOP_RETURN: IDASolve succeeded by reaching the specified stopping point.",
            0: "IDA_SUCCESS: Successful function return.",
            -1: "IDA_TOO_MUCH_WORK: The solver took mxstep internal steps but could not reach tout.",
            -2: "IDA_TOO_MUCH_ACC: The solver could not satisfy the accuracy demanded by the user for some internal step.",
            -3: "IDA_ERR_FAIL: Error test failures occurred too many times during one internal time step or minimum step size was reached.",
            -4: "IDA_CONV_FAIL: Convergence test failures occurred too many times during one internal time step or minimum step size was reached.",
            -5: "IDA_LINIT_FAIL: The linear solver's initialization function failed.",
            -6: "IDA_LSETUP_FAIL: The linear solver's setup function failed in an unrecoverable manner.",
            -7: "IDA_LSOLVE_FAIL: The linear solver's solve function failed in an unrecoverable manner.",
            -8: "IDA_RES_FAIL: The user-provided residual function failed in an unrecoverable manner.",
            -9: "IDA_REP_RES_FAIL: The user-provided residual function repeatedly returned a recoverable error flag, but the solver was unable to recover.",
            -10: "IDA_RTFUNC_FAIL: The rootfinding function failed in an unrecoverable manner.",
            -11: "IDA_CONSTR_FAIL: The inequality constraints were violated and the solver was unable to recover.",
            -12: "IDA_FIRST_RES_FAIL: The user-provided residual function failed recoverably on the first call.",
            -13: "IDA_LINESEARCH_FAIL: The line search failed.",
            -14: "IDA_NO_RECOVERY: The residual function, linear solver setup function, or linear solver solve function had a recoverable failure, but IDACalcIC could not recover.",
            -15: "IDA_NLS_INIT_FAIL: The nonlinear solver's init routine failed.",
            -16: "IDA_NLS_SETUP_FAIL: The nonlinear solver's setup routine failed.",
            -20: "IDA_MEM_NULL: The ida mem argument was NULL.",
            -21: "IDA_MEM_FAIL: A memory allocation failed.",
            -22: "IDA_ILL_INPUT: One of the function inputs is illegal.",
            -23: "IDA_NO_MALLOC: The ida memory was not allocated by a call to IDAInit.",
            -24: "IDA_BAD_EWT: Zero value of some error weight component.",
            -25: "IDA_BAD_K: The k-th derivative is not available.",
            -26: "IDA_BAD_T: The time t is outside the last step taken.",
            -27: "IDA_BAD_DKY: The vector argument where derivative should be stored is NULL.",
        }

        flag_unknown = "Unknown IDA flag."

        return flags.get(flag, flag_unknown)
