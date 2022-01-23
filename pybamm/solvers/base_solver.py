#
# Base solver class
#
import copy
import itertools
from scipy.sparse import block_diag
import multiprocessing as mp
import numbers
import sys
import warnings

import casadi
import numpy as np

import pybamm
from pybamm.expression_tree.binary_operators import _Heaviside


class BaseSolver(object):
    """Solve a discretised model.

    Parameters
    ----------
    method : str, optional
        The method to use for integration, specific to each solver
    rtol : float, optional
        The relative tolerance for the solver (default is 1e-6).
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
        The tolerance to assert whether extrapolation occurs or not. Default is 0.
    """

    def __init__(
        self,
        method=None,
        rtol=1e-6,
        atol=1e-6,
        root_method=None,
        root_tol=1e-6,
        extrap_tol=0,
        max_steps="deprecated",
    ):
        self.method = method
        self.rtol = rtol
        self.atol = atol
        self.root_tol = root_tol
        self.root_method = root_method
        self.extrap_tol = extrap_tol
        if max_steps != "deprecated":
            raise ValueError(
                "max_steps has been deprecated, and should be set using the "
                "solver-specific extra-options dictionaries instead"
            )
        self.models_set_up = {}

        # Defaults, can be overwritten by specific solver
        self.name = "Base solver"
        self.ode_solver = False
        self.algebraic_solver = False

    @property
    def root_method(self):
        return self._root_method

    @root_method.setter
    def root_method(self, method):
        if method == "casadi":
            method = pybamm.CasadiAlgebraicSolver(self.root_tol)
        elif isinstance(method, str):
            method = pybamm.AlgebraicSolver(method, self.root_tol)
        elif not (
            method is None
            or (
                isinstance(method, pybamm.BaseSolver)
                and method.algebraic_solver is True
            )
        ):
            raise pybamm.SolverError("Root method must be an algebraic solver")
        self._root_method = method

    def copy(self):
        """Returns a copy of the solver"""
        new_solver = copy.copy(self)
        # clear models_set_up
        new_solver.models_set_up = {}
        return new_solver

    def set_up(self, model, inputs=None, t_eval=None, ics_only=False):
        """Unpack model, perform checks, and calculate jacobian.

        Parameters
        ----------
        model : :class:`pybamm.BaseModel`
            The model whose solution to calculate. Must have attributes rhs and
            initial_conditions
        inputs : dict, optional
            Any input parameters to pass to the model when solving
        t_eval : numeric type, optional
            The times (in seconds) at which to compute the solution
        """
        pybamm.logger.info("Start solver set-up")

        # Check model.algebraic for ode solvers
        if self.ode_solver is True and len(model.algebraic) > 0:
            raise pybamm.SolverError(
                "Cannot use ODE solver '{}' to solve DAE model".format(self.name)
            )
        # Check model.rhs for algebraic solvers
        if self.algebraic_solver is True and len(model.rhs) > 0:
            raise pybamm.SolverError(
                """Cannot use algebraic solver to solve model with time derivatives"""
            )
        # casadi solver won't allow solving algebraic model so we have to raise an
        # error here
        if isinstance(self, pybamm.CasadiSolver) and len(model.rhs) == 0:
            raise pybamm.SolverError(
                "Cannot use CasadiSolver to solve algebraic model, "
                "use CasadiAlgebraicSolver instead"
            )
        # Discretise model if it isn't already discretised
        # This only works with purely 0D models, as otherwise the mesh and spatial
        # method should be specified by the user
        if model.is_discretised is False:
            try:
                disc = pybamm.Discretisation()
                disc.process_model(model)
            except pybamm.DiscretisationError as e:
                raise pybamm.DiscretisationError(
                    "Cannot automatically discretise model, "
                    "model should be discretised before solving ({})".format(e)
                )

        inputs = inputs or {}

        # Set model timescale
        if not isinstance(model.timescale, pybamm.Scalar):
            raise ValueError("model.timescale must be a scalar")

        model.timescale_eval = model.timescale.evaluate()
        # Set model lengthscales
        model.length_scales_eval = {
            domain: scale.evaluate(inputs=inputs)
            for domain, scale in model.length_scales.items()
        }
        if (
            isinstance(self, (pybamm.CasadiSolver, pybamm.CasadiAlgebraicSolver))
        ) and model.convert_to_format != "casadi":
            pybamm.logger.warning(
                "Converting {} to CasADi for solving with CasADi solver".format(
                    model.name
                )
            )
            model.convert_to_format = "casadi"
        if (
            isinstance(self.root_method, pybamm.CasadiAlgebraicSolver)
            and model.convert_to_format != "casadi"
        ):
            pybamm.logger.warning(
                "Converting {} to CasADi for calculating ICs with CasADi".format(
                    model.name
                )
            )
            model.convert_to_format = "casadi"

        # find all the input parameters in the model
        input_parameters = {}
        for equation in [
            model.concatenated_rhs,
            model.concatenated_algebraic,
            model.concatenated_initial_conditions,
        ]:
            input_parameters.update(
                {
                    symbol._id: symbol
                    for symbol in equation.pre_order()
                    if isinstance(symbol, pybamm.InputParameter)
                }
            )

        # set default calculate sensitivities on model
        if not hasattr(model, "calculate_sensitivities"):
            model.calculate_sensitivities = []

        # see if we need to form the explicit sensitivity equations
        calculate_sensitivities_explicit = False
        if model.calculate_sensitivities and not isinstance(self, pybamm.IDAKLUSolver):
            calculate_sensitivities_explicit = True

        # if we are calculating sensitivities explicitly then the number of
        # equations will change
        if calculate_sensitivities_explicit:
            num_parameters = 0
            for name in model.calculate_sensitivities:
                # if not a number, assume its a vector
                if isinstance(inputs[name], numbers.Number):
                    num_parameters += 1
                else:
                    num_parameters += len(inputs[name])
            model.len_rhs_sens = model.len_rhs * num_parameters
            model.len_alg_sens = model.len_alg * num_parameters

        if model.convert_to_format != "casadi":
            # Create Jacobian from concatenated rhs and algebraic
            y = pybamm.StateVector(slice(0, model.len_rhs_and_alg))
            # set up Jacobian object, for re-use of dict
            jacobian = pybamm.Jacobian()

        else:
            # Convert model attributes to casadi
            t_casadi = casadi.MX.sym("t")
            # Create the symbolic state vectors
            y_diff = casadi.MX.sym("y_diff", model.len_rhs)
            y_alg = casadi.MX.sym("y_alg", model.len_alg)
            y_casadi = casadi.vertcat(y_diff, y_alg)
            p_casadi = {}
            for name, value in inputs.items():
                if isinstance(value, numbers.Number):
                    p_casadi[name] = casadi.MX.sym(name)
                else:
                    p_casadi[name] = casadi.MX.sym(name, value.shape[0])
            p_casadi_stacked = casadi.vertcat(*[p for p in p_casadi.values()])
            # sensitivity vectors
            if calculate_sensitivities_explicit:
                pS_casadi_stacked = casadi.vertcat(
                    *[p_casadi[name] for name in model.calculate_sensitivities]
                )
                S_x = casadi.MX.sym("S_x", model.len_rhs_sens)
                S_z = casadi.MX.sym("S_z", model.len_alg_sens)
                y_and_S = casadi.vertcat(y_diff, S_x, y_alg, S_z)
            else:
                y_and_S = y_casadi

        def process(func, name, use_jacobian=None):
            def report(string):
                # don't log event conversion
                if "event" not in string:
                    pybamm.logger.verbose(string)

            if use_jacobian is None:
                use_jacobian = model.use_jacobian

            if model.convert_to_format == "jax":
                report(f"Converting {name} to jax")
                func = pybamm.EvaluatorJax(func)
                jacp = None
                if model.calculate_sensitivities:
                    report(
                        (
                            f"Calculating sensitivities for {name} with respect "
                            f"to parameters {model.calculate_sensitivities} using jax"
                        )
                    )
                    jacp = func.get_sensitivities()
                    jacp = jacp.evaluate
                if use_jacobian:
                    report(f"Calculating jacobian for {name} using jax")
                    jac = func.get_jacobian()
                    jac = jac.evaluate
                else:
                    jac = None

                func = func.evaluate

            elif model.convert_to_format != "casadi":
                # Process with pybamm functions, optionally converting
                # to python evaluator
                if model.calculate_sensitivities:
                    report(
                        (
                            f"Calculating sensitivities for {name} with respect "
                            f"to parameters {model.calculate_sensitivities}"
                        )
                    )
                    jacp_dict = {
                        p: func.diff(pybamm.InputParameter(p))
                        for p in model.calculate_sensitivities
                    }
                    if model.convert_to_format == "python":
                        report(f"Converting sensitivities for {name} to python")
                        jacp_dict = {
                            p: pybamm.EvaluatorPython(jacp)
                            for p, jacp in jacp_dict.items()
                        }

                    # jacp should be a function that returns a dict of sensitivities
                    def jacp(*args, **kwargs):
                        return {
                            k: v.evaluate(*args, **kwargs) for k, v in jacp_dict.items()
                        }

                else:
                    jacp = None

                if use_jacobian:
                    report(f"Calculating jacobian for {name}")
                    jac = jacobian.jac(func, y)
                    if model.convert_to_format == "python":
                        report(f"Converting jacobian for {name} to python")
                        jac = pybamm.EvaluatorPython(jac)
                    jac = jac.evaluate
                else:
                    jac = None

                if model.convert_to_format == "python":
                    report(f"Converting {name} to python")
                    func = pybamm.EvaluatorPython(func)

                func = func.evaluate

            else:
                # Process with CasADi
                report(f"Converting {name} to CasADi")
                func = func.to_casadi(t_casadi, y_casadi, inputs=p_casadi)
                # Add sensitivity vectors to the rhs and algebraic equations
                jacp = None
                if calculate_sensitivities_explicit:
                    # The formulation is as per Park, S., Kato, D., Gima, Z., Klein, R.,
                    # & Moura, S. (2018).  Optimal experimental design for
                    # parameterization of an electrochemical lithium-ion battery model.
                    # Journal of The Electrochemical Society, 165(7), A1309.". See #1100
                    # for details
                    if name == "RHS" and model.len_rhs > 0:
                        report(
                            "Creating explicit forward sensitivity equations "
                            "for rhs using CasADi"
                        )
                        df_dx = casadi.jacobian(func, y_diff)
                        df_dp = casadi.jacobian(func, pS_casadi_stacked)
                        S_x_mat = S_x.reshape(
                            (model.len_rhs, pS_casadi_stacked.shape[0])
                        )
                        if model.len_alg == 0:
                            S_rhs = (df_dx @ S_x_mat + df_dp).reshape((-1, 1))
                        else:
                            df_dz = casadi.jacobian(func, y_alg)
                            S_z_mat = S_z.reshape(
                                (model.len_alg, pS_casadi_stacked.shape[0])
                            )
                            S_rhs = (df_dx @ S_x_mat + df_dz @ S_z_mat + df_dp).reshape(
                                (-1, 1)
                            )
                        func = casadi.vertcat(func, S_rhs)
                    if name == "algebraic" and model.len_alg > 0:
                        report(
                            "Creating explicit forward sensitivity equations "
                            "for algebraic using CasADi"
                        )
                        dg_dz = casadi.jacobian(func, y_alg)
                        dg_dp = casadi.jacobian(func, pS_casadi_stacked)
                        S_z_mat = S_z.reshape(
                            (model.len_alg, pS_casadi_stacked.shape[0])
                        )
                        if model.len_rhs == 0:
                            S_alg = (dg_dz @ S_z_mat + dg_dp).reshape((-1, 1))
                        else:
                            dg_dx = casadi.jacobian(func, y_diff)
                            S_x_mat = S_x.reshape(
                                (model.len_rhs, pS_casadi_stacked.shape[0])
                            )
                            S_alg = (dg_dx @ S_x_mat + dg_dz @ S_z_mat + dg_dp).reshape(
                                (-1, 1)
                            )
                        func = casadi.vertcat(func, S_alg)
                    if name == "initial_conditions":
                        if model.len_rhs == 0 or model.len_alg == 0:
                            S_0 = casadi.jacobian(func, pS_casadi_stacked).reshape(
                                (-1, 1)
                            )
                            func = casadi.vertcat(func, S_0)
                        else:
                            x0 = func[: model.len_rhs]
                            z0 = func[model.len_rhs :]
                            Sx_0 = casadi.jacobian(x0, pS_casadi_stacked).reshape(
                                (-1, 1)
                            )
                            Sz_0 = casadi.jacobian(z0, pS_casadi_stacked).reshape(
                                (-1, 1)
                            )
                            func = casadi.vertcat(x0, Sx_0, z0, Sz_0)
                elif model.calculate_sensitivities:
                    report(
                        (
                            f"Calculating sensitivities for {name} with respect "
                            f"to parameters {model.calculate_sensitivities} using "
                            "CasADi"
                        )
                    )
                    jacp_dict = {}
                    for pname in model.calculate_sensitivities:
                        p_diff = casadi.jacobian(func, p_casadi[pname])
                        jacp_dict[pname] = casadi.Function(
                            name, [t_casadi, y_casadi, p_casadi_stacked], [p_diff]
                        )
                    # jacp should be a function that returns a dict of sensitivities

                    def jacp(*args, **kwargs):
                        return {k: v(*args, **kwargs) for k, v in jacp_dict.items()}

                if use_jacobian:
                    report(f"Calculating jacobian for {name} using CasADi")
                    jac_casadi = casadi.jacobian(func, y_and_S)
                    jac = casadi.Function(
                        name, [t_casadi, y_and_S, p_casadi_stacked], [jac_casadi]
                    )
                else:
                    jac = None

                func = casadi.Function(
                    name, [t_casadi, y_and_S, p_casadi_stacked], [func]
                )
            if name == "residuals":
                func_call = Residuals(func, name, model)
            else:
                func_call = SolverCallable(func, name, model)
            if jac is not None:
                jac_call = SolverCallable(jac, name + "_jac", model)
            else:
                jac_call = None
            if jacp is not None:
                jacp_call = SensitivityCallable(
                    jacp,
                    name + "_sensitivity_wrt_inputs",
                    model,
                    model.convert_to_format,
                )
            else:
                jacp_call = None
            return func, func_call, jac_call, jacp_call

        # Process initial conditions
        initial_conditions = process(
            model.concatenated_initial_conditions,
            "initial_conditions",
            use_jacobian=False,
        )[0]
        init_eval = InitialConditions(initial_conditions, model)

        if calculate_sensitivities_explicit:
            y0_total_size = (
                model.len_rhs + model.len_rhs_sens + model.len_alg + model.len_alg_sens
            )
            init_eval.y_dummy = np.zeros((y0_total_size, 1))
        else:
            init_eval.y_dummy = np.zeros((model.len_rhs_and_alg, 1))

        # Calculate initial conditions
        model.y0 = init_eval(inputs)
        model.init_eval = init_eval

        if not ics_only:
            # Check for heaviside and modulo functions in rhs and algebraic and add
            # discontinuity events if these exist.
            # Note: only checks for the case of t < X, t <= X, X < t, or X <= t,
            # but also accounts for the fact that t might be dimensional
            # Only do this for DAE models as ODE models can deal with discontinuities
            # fine
            if len(model.algebraic) > 0:
                for symbol in itertools.chain(
                    model.concatenated_rhs.pre_order(),
                    model.concatenated_algebraic.pre_order(),
                ):
                    if isinstance(symbol, _Heaviside):
                        found_t = False
                        # Dimensionless
                        if symbol.right.id == pybamm.t.id:
                            expr = symbol.left
                            found_t = True
                        elif symbol.left.id == pybamm.t.id:
                            expr = symbol.right
                            found_t = True
                        # Dimensional
                        elif symbol.right.id == (pybamm.t * model.timescale_eval).id:
                            expr = symbol.left / symbol.right.right
                            found_t = True
                        elif symbol.left.id == (pybamm.t * model.timescale_eval).id:
                            expr = symbol.right / symbol.left.right
                            found_t = True

                        # Update the events if the heaviside function depended on t
                        if found_t:
                            model.events.append(
                                pybamm.Event(
                                    str(symbol), expr, pybamm.EventType.DISCONTINUITY
                                )
                            )
                    elif isinstance(symbol, pybamm.Modulo):
                        found_t = False
                        # Dimensionless
                        if symbol.left.id == pybamm.t.id:
                            expr = symbol.right
                            found_t = True
                        # Dimensional
                        elif symbol.left.id == (pybamm.t * model.timescale_eval).id:
                            expr = symbol.right / symbol.left.right
                            found_t = True

                        # Update the events if the modulo function depended on t
                        if found_t:
                            if t_eval is None:
                                N_events = 200
                            else:
                                N_events = t_eval[-1] // expr.value

                            for i in np.arange(N_events):
                                model.events.append(
                                    pybamm.Event(
                                        str(symbol),
                                        expr * pybamm.Scalar(i + 1),
                                        pybamm.EventType.DISCONTINUITY,
                                    )
                                )

            # Process rhs, algebraic and event expressions
            rhs, rhs_eval, jac_rhs, jacp_rhs = process(model.concatenated_rhs, "RHS")
            algebraic, algebraic_eval, jac_algebraic, jacp_algebraic = process(
                model.concatenated_algebraic, "algebraic"
            )
            casadi_terminate_events = []
            terminate_events_eval = []
            interpolant_extrapolation_events_eval = []
            discontinuity_events_eval = []
            for n, event in enumerate(model.events):
                if event.event_type == pybamm.EventType.DISCONTINUITY:
                    # discontinuity events are evaluated before the solver is called,
                    # so don't need to process them
                    discontinuity_events_eval.append(event)
                elif event.event_type == pybamm.EventType.SWITCH:
                    if (
                        isinstance(self, pybamm.CasadiSolver)
                        and self.mode == "fast with events"
                        and model.algebraic != {}
                    ):
                        # Save some events to casadi_terminate_events for the 'fast with
                        # events' mode of the casadi solver
                        # We only need to do this if the model is a DAE model
                        # see #1082
                        k = 20
                        init_sign = float(
                            np.sign(event.evaluate(0, model.y0.full(), inputs=inputs))
                        )
                        # We create a sigmoid for each event which will multiply the
                        # rhs. Doing * 2 - 1 ensures that when the event is crossed,
                        # the sigmoid is zero. Hence the rhs is zero and the solution
                        # stays constant for the rest of the simulation period
                        # We can then cut off the part after the event was crossed
                        event_sigmoid = (
                            pybamm.sigmoid(0, init_sign * event.expression, k) * 2 - 1
                        )
                        event_casadi = process(
                            event_sigmoid, f"event_{n}", use_jacobian=False
                        )[0]
                        # use the actual casadi object as this will go into the rhs
                        casadi_terminate_events.append(event_casadi)
                else:
                    # use the function call
                    event_eval = process(
                        event.expression, f"event_{n}", use_jacobian=False
                    )[1]
                    if event.event_type == pybamm.EventType.TERMINATION:
                        terminate_events_eval.append(event_eval)
                    elif event.event_type == pybamm.EventType.INTERPOLANT_EXTRAPOLATION:
                        interpolant_extrapolation_events_eval.append(event_eval)

            # Add the solver attributes
            model.rhs_eval = rhs_eval
            model.algebraic_eval = algebraic_eval
            model.jac_algebraic_eval = jac_algebraic
            model.casadi_terminate_events = casadi_terminate_events
            model.terminate_events_eval = terminate_events_eval
            model.discontinuity_events_eval = discontinuity_events_eval
            model.interpolant_extrapolation_events_eval = (
                interpolant_extrapolation_events_eval
            )

            # if we have changed the equations to include the explicit sensitivity
            # equations, then we also need to update the mass matrix and bounds
            if calculate_sensitivities_explicit:
                if model.len_rhs != 0:
                    n_inputs = model.len_rhs_sens // model.len_rhs
                elif model.len_alg != 0:
                    n_inputs = model.len_alg_sens // model.len_alg
                if model.bounds[0].shape[0] == model.len_rhs_and_alg:
                    model.bounds = (
                        np.repeat(model.bounds[0], n_inputs + 1),
                        np.repeat(model.bounds[1], n_inputs + 1),
                    )
                if (
                    model.mass_matrix is not None
                    and model.mass_matrix.shape[0] == model.len_rhs_and_alg
                ):

                    if model.mass_matrix_inv is not None:
                        model.mass_matrix_inv = pybamm.Matrix(
                            block_diag(
                                [model.mass_matrix_inv.entries] * (n_inputs + 1),
                                format="csr",
                            )
                        )
                    model.mass_matrix = pybamm.Matrix(
                        block_diag(
                            [model.mass_matrix.entries] * (n_inputs + 1), format="csr"
                        )
                    )
            else:
                # take care if calculate_sensitivites used then not used
                if model.bounds[0].shape[0] > model.len_rhs_and_alg:
                    model.bounds = (
                        model.bounds[0][: model.len_rhs_and_alg],
                        model.bounds[1][: model.len_rhs_and_alg],
                    )
                if (
                    model.mass_matrix is not None
                    and model.mass_matrix.shape[0] > model.len_rhs_and_alg
                ):
                    if model.mass_matrix_inv is not None:
                        model.mass_matrix_inv = pybamm.Matrix(
                            model.mass_matrix_inv.entries[
                                : model.len_rhs, : model.len_rhs
                            ]
                        )
                    model.mass_matrix = pybamm.Matrix(
                        model.mass_matrix.entries[
                            : model.len_rhs_and_alg, : model.len_rhs_and_alg
                        ]
                    )

            # Save CasADi functions for the CasADi solver
            # Note: when we pass to casadi the ode part of the problem must be in
            # explicit form so we pre-multiply by the inverse of the mass matrix
            if isinstance(self.root_method, pybamm.CasadiAlgebraicSolver) or isinstance(
                self, (pybamm.CasadiSolver, pybamm.CasadiAlgebraicSolver)
            ):
                # can use DAE solver to solve model with algebraic equations only
                if len(model.rhs) > 0:
                    mass_matrix_inv = casadi.MX(model.mass_matrix_inv.entries)
                    explicit_rhs = mass_matrix_inv @ rhs(
                        t_casadi, y_and_S, p_casadi_stacked
                    )
                    model.casadi_rhs = casadi.Function(
                        "rhs", [t_casadi, y_and_S, p_casadi_stacked], [explicit_rhs]
                    )
                model.casadi_algebraic = algebraic
                model.casadi_sensitivities_rhs = jacp_rhs
                model.casadi_sensitivities_algebraic = jacp_algebraic
            if len(model.rhs) == 0:
                # No rhs equations: residuals is algebraic only
                model.residuals_eval = Residuals(algebraic, "residuals", model)
                model.jacobian_eval = jac_algebraic
                model.sensitivities_eval = jacp_algebraic
            elif len(model.algebraic) == 0:
                # No algebraic equations: residuals is rhs only
                model.residuals_eval = Residuals(rhs, "residuals", model)
                model.jacobian_eval = jac_rhs
                model.sensitivities_eval = jacp_rhs
            # Calculate consistent initial conditions for the algebraic equations
            else:
                all_states = pybamm.NumpyConcatenation(
                    model.concatenated_rhs, model.concatenated_algebraic
                )
                # Process again, uses caching so should be quick
                residuals_eval, jacobian_eval, jacobian_wrtp_eval = process(
                    all_states, "residuals"
                )[1:]
                model.residuals_eval = residuals_eval
                model.jacobian_eval = jacobian_eval
                model.sensitivities_eval = jacobian_wrtp_eval

        pybamm.logger.info("Finish solver set-up")

    def _set_initial_conditions(self, model, inputs, update_rhs):
        """
        Set initial conditions for the model. This is skipped if the solver is an
        algebraic solver (since this would make the algebraic solver redundant), and if
        the model doesn't have any algebraic equations (since there are no initial
        conditions to be calculated in this case).

        Parameters
        ----------
        model : :class:`pybamm.BaseModel`
            The model for which to calculate initial conditions.
        inputs : dict
            Any input parameters to pass to the model when solving
        update_rhs : bool
            Whether to update the rhs. True for 'solve', False for 'step'.

        """

        if self.algebraic_solver is True:
            # Don't update model.y0
            return None
        elif len(model.algebraic) == 0:
            if update_rhs is True:
                # Recalculate initial conditions for the rhs equations
                y0 = model.init_eval(inputs)
            else:
                # Don't update model.y0
                return None
        else:
            if update_rhs is True:
                # Recalculate initial conditions for the rhs equations
                y0_from_inputs = model.init_eval(inputs)
                # Reuse old solution for algebraic equations
                y0_from_model = model.y0
                len_rhs = model.len_rhs
                # update model.y0, which is used for initialising the algebraic solver
                if len_rhs == 0:
                    model.y0 = y0_from_model
                else:
                    model.y0 = casadi.vertcat(
                        y0_from_inputs[:len_rhs], y0_from_model[len_rhs:]
                    )
            y0 = self.calculate_consistent_state(model, 0, inputs)
        # Make y0 a function of inputs if doing symbolic with casadi
        model.y0 = y0

    def calculate_consistent_state(self, model, time=0, inputs=None):
        """
        Calculate consistent state for the algebraic equations through
        root-finding. model.y0 is used as the initial guess for rootfinding

        Parameters
        ----------
        model : :class:`pybamm.BaseModel`
            The model for which to calculate initial conditions.
        time : float
            The time at which to calculate the states
        inputs_dict : dict, optional
            Any input parameters to pass to the model when solving

        Returns
        -------
        y0_consistent : array-like, same shape as y0_guess
            Initial conditions that are consistent with the algebraic equations (roots
            of the algebraic equations). If self.root_method == None then returns
            model.y0.
        """
        pybamm.logger.debug("Start calculating consistent states")
        if self.root_method is None:
            return model.y0
        try:
            root_sol = self.root_method._integrate(model, np.array([time]), inputs)
        except pybamm.SolverError as e:
            raise pybamm.SolverError(
                "Could not find consistent states: {}".format(e.args[0])
            )
        pybamm.logger.debug("Found consistent states")

        y0 = root_sol.all_ys[0]
        if isinstance(y0, np.ndarray):
            y0 = y0.flatten()
        return y0

    def solve(
        self,
        model,
        t_eval=None,
        external_variables=None,
        inputs=None,
        initial_conditions=None,
        nproc=None,
        calculate_sensitivities=False,
    ):
        """
        Execute the solver setup and calculate the solution of the model at
        specified times.

        Parameters
        ----------
        model : :class:`pybamm.BaseModel`
            The model whose solution to calculate. Must have attributes rhs and
            initial_conditions
        t_eval : numeric type
            The times (in seconds) at which to compute the solution
        external_variables : dict
            A dictionary of external variables and their corresponding
            values at the current time
        inputs : dict or list, optional
            A dictionary or list of dictionaries describing any input parameters to
            pass to the model when solving
        initial_conditions : :class:`pybamm.Symbol`, optional
            Initial conditions to use when solving the model. If None (default),
            `model.concatenated_initial_conditions` is used. Otherwise, must be a symbol
            of size `len(model.rhs) + len(model.algebraic)`.
        nproc : int, optional
            Number of processes to use when solving for more than one set of input
            parameters. Defaults to value returned by "os.cpu_count()".
        calculate_sensitivites : list of str or bool
            If true, solver calculates sensitivities of all input parameters.
            If only a subset of sensitivities are required, can also pass a
            list of input parameter names

        Returns
        -------
        :class:`pybamm.Solution` or list of :class:`pybamm.Solution` objects.
             If type of `inputs` is `list`, return a list of corresponding
             :class:`pybamm.Solution` objects.

        Raises
        ------
        :class:`pybamm.ModelError`
            If an empty model is passed (`model.rhs = {}` and `model.algebraic={}` and
            `model.variables = {}`)

        """
        pybamm.logger.info("Start solving {} with {}".format(model.name, self.name))

        # get a list-only version of calculate_sensitivities
        if isinstance(calculate_sensitivities, bool):
            if calculate_sensitivities:
                calculate_sensitivities_list = [p for p in inputs.keys()]
            else:
                calculate_sensitivities_list = []
        else:
            calculate_sensitivities_list = calculate_sensitivities

        # Make sure model isn't empty
        if len(model.rhs) == 0 and len(model.algebraic) == 0:
            if not isinstance(self, pybamm.DummySolver):
                raise pybamm.ModelError(
                    "Cannot solve empty model, use `pybamm.DummySolver` instead"
                )

        # t_eval can only be None if the solver is an algebraic solver. In that case
        # set it to 0
        if t_eval is None:
            if self.algebraic_solver is True:
                t_eval = np.array([0])
            else:
                raise ValueError("t_eval cannot be None")

        # If t_eval is provided as [t0, tf] return the solution at 100 points
        elif isinstance(t_eval, list):
            if len(t_eval) == 1 and self.algebraic_solver is True:
                pass
            elif len(t_eval) != 2:
                raise pybamm.SolverError(
                    "'t_eval' can be provided as an array of times at which to "
                    "return the solution, or as a list [t0, tf] where t0 is the "
                    "initial time and tf is the final time, but has been provided "
                    "as a list of length {}.".format(len(t_eval))
                )
            else:
                t_eval = np.linspace(t_eval[0], t_eval[-1], 100)

        # Make sure t_eval is monotonic
        if (np.diff(t_eval) < 0).any():
            raise pybamm.SolverError("t_eval must increase monotonically")

        # Set up external variables and inputs
        #
        # Argument "inputs" can be either a list of input dicts or
        # a single dict. The remaining of this function is only working
        # with variable "input_list", which is a list of dictionaries.
        # If "inputs" is a single dict, "inputs_list" is a list of only one dict.
        inputs_list = inputs if isinstance(inputs, list) else [inputs]
        ext_and_inputs_list = [
            self._set_up_ext_and_inputs(
                model, external_variables, inputs, calculate_sensitivities_list
            )
            for inputs in inputs_list
        ]

        # Cannot use multiprocessing with model in "jax" format
        if (len(inputs_list) > 1) and model.convert_to_format == "jax":
            raise pybamm.SolverError(
                "Cannot solve list of inputs with multiprocessing "
                'when model in format "jax".'
            )

        # Check that calculate_sensitivites have not been updated
        calculate_sensitivities_list.sort()
        if not hasattr(model, "calculate_sensitivities"):
            model.calculate_sensitivities = []
        model.calculate_sensitivities.sort()
        if calculate_sensitivities_list != model.calculate_sensitivities:
            self.models_set_up.pop(model, None)
            # CasadiSolver caches its integrators using model, so delete this too
            if isinstance(self, pybamm.CasadiSolver):
                self.integrators.pop(model, None)

        # save sensitivity parameters so we can identify them later on
        # (FYI: this is used in the Solution class)
        model.calculate_sensitivities = calculate_sensitivities_list

        # Set up (if not done already)
        timer = pybamm.Timer()
        if model not in self.models_set_up:
            # It is assumed that when len(inputs_list) > 1, model set
            # up (initial condition, time-scale and length-scale) does
            # not depend on input parameters. Thefore only `ext_and_inputs[0]`
            # is passed to `set_up`.
            # See https://github.com/pybamm-team/PyBaMM/pull/1261
            self.set_up(model, ext_and_inputs_list[0], t_eval)
            self.models_set_up.update(
                {model: {"initial conditions": model.concatenated_initial_conditions}}
            )
        else:
            ics_set_up = self.models_set_up[model]["initial conditions"]
            # Check that initial conditions have not been updated
            if ics_set_up.id != model.concatenated_initial_conditions.id:
                # If the new initial conditions are different, set up again
                self.set_up(model, ext_and_inputs_list[0], t_eval, ics_only=True)
                self.models_set_up[model][
                    "initial conditions"
                ] = model.concatenated_initial_conditions

        set_up_time = timer.time()
        timer.reset()

        # (Re-)calculate consistent initial conditions
        # Assuming initial conditions do not depend on input parameters
        # when len(inputs_list) > 1, only `ext_and_inputs_list[0]`
        # is passed to `_set_initial_conditions`.
        # See https://github.com/pybamm-team/PyBaMM/pull/1261
        if len(inputs_list) > 1:
            all_inputs_names = set(
                itertools.chain.from_iterable(
                    [ext_and_inputs.keys() for ext_and_inputs in ext_and_inputs_list]
                )
            )
            initial_conditions_node_names = set(
                [it.name for it in model.concatenated_initial_conditions.pre_order()]
            )
            if all_inputs_names.issubset(initial_conditions_node_names):
                raise pybamm.SolverError(
                    "Input parameters cannot appear in expression "
                    "for initial conditions."
                )

        self._set_initial_conditions(model, ext_and_inputs_list[0], update_rhs=True)

        # Non-dimensionalise time
        t_eval_dimensionless = t_eval / model.timescale_eval

        # Calculate discontinuities
        discontinuities = [
            # Assuming that discontinuities do not depend on
            # input parameters when len(input_list) > 1, only
            # `input_list[0]` is passed to `evaluate`.
            # See https://github.com/pybamm-team/PyBaMM/pull/1261
            event.expression.evaluate(inputs=inputs_list[0])
            for event in model.discontinuity_events_eval
        ]

        # make sure they are increasing in time
        discontinuities = sorted(discontinuities)

        # remove any identical discontinuities
        discontinuities = [
            v
            for i, v in enumerate(discontinuities)
            if (
                i == len(discontinuities) - 1
                or discontinuities[i] < discontinuities[i + 1]
            )
            and v > 0
        ]

        # remove any discontinuities after end of t_eval
        discontinuities = [v for v in discontinuities if v < t_eval_dimensionless[-1]]

        if len(discontinuities) > 0:
            pybamm.logger.verbose(
                "Discontinuity events found at t = {}".format(discontinuities)
            )
            if isinstance(inputs, list):
                raise pybamm.SolverError(
                    "Cannot solve for a list of input parameters"
                    " sets with discontinuities"
                )
        else:
            pybamm.logger.verbose("No discontinuity events found")

        # insert time points around discontinuities in t_eval
        # keep track of sub sections to integrate by storing start and end indices
        start_indices = [0]
        end_indices = []
        eps = sys.float_info.epsilon
        for dtime in discontinuities:
            dindex = np.searchsorted(t_eval_dimensionless, dtime, side="left")
            end_indices.append(dindex + 1)
            start_indices.append(dindex + 1)
            if dtime - eps < t_eval_dimensionless[dindex] < dtime + eps:
                t_eval_dimensionless[dindex] += eps
                t_eval_dimensionless = np.insert(
                    t_eval_dimensionless, dindex, dtime - eps
                )
            else:
                t_eval_dimensionless = np.insert(
                    t_eval_dimensionless, dindex, [dtime - eps, dtime + eps]
                )
        end_indices.append(len(t_eval_dimensionless))

        # Integrate separately over each time segment and accumulate into the solution
        # object, restarting the solver at each discontinuity (and recalculating a
        # consistent state afterwards if a DAE)
        old_y0 = model.y0
        solutions = None
        for start_index, end_index in zip(start_indices, end_indices):
            pybamm.logger.verbose(
                "Calling solver for {} < t < {}".format(
                    t_eval_dimensionless[start_index] * model.timescale_eval,
                    t_eval_dimensionless[end_index - 1] * model.timescale_eval,
                )
            )
            ninputs = len(ext_and_inputs_list)
            if ninputs == 1:
                new_solution = self._integrate(
                    model,
                    t_eval_dimensionless[start_index:end_index],
                    ext_and_inputs_list[0],
                )
                new_solutions = [new_solution]
            else:
                with mp.Pool(processes=nproc) as p:
                    new_solutions = p.starmap(
                        self._integrate,
                        zip(
                            [model] * ninputs,
                            [t_eval_dimensionless[start_index:end_index]] * ninputs,
                            ext_and_inputs_list,
                        ),
                    )
                    p.close()
                    p.join()
            # Setting the solve time for each segment.
            # pybamm.Solution.__add__ assumes attribute solve_time.
            solve_time = timer.time()
            for sol in new_solutions:
                sol.solve_time = solve_time
            if start_index == start_indices[0]:
                solutions = [sol for sol in new_solutions]
            else:
                for i, new_solution in enumerate(new_solutions):
                    solutions[i] = solutions[i] + new_solution

            if solutions[0].termination != "final time":
                break

            if end_index != len(t_eval_dimensionless):
                # setup for next integration subsection
                last_state = solutions[0].y[:, -1]
                # update y0 (for DAE solvers, this updates the initial guess for the
                # rootfinder)
                model.y0 = last_state
                if len(model.algebraic) > 0:
                    model.y0 = self.calculate_consistent_state(
                        model, t_eval_dimensionless[end_index], ext_and_inputs_list[0]
                    )
        solve_time = timer.time()

        for i, solution in enumerate(solutions):
            # Check if extrapolation occurred
            extrapolation = self.check_extrapolation(solution, model.events)
            if extrapolation:
                warnings.warn(
                    "While solving {} extrapolation occurred for {}".format(
                        model.name, extrapolation
                    ),
                    pybamm.SolverWarning,
                )
            # Identify the event that caused termination and update the solution to
            # include the event time and state
            solutions[i], termination = self.get_termination_reason(
                solution, model.events
            )
            # Assign times
            solutions[i].set_up_time = set_up_time
            # all solutions get the same solve time, but their integration time
            # will be different (see https://github.com/pybamm-team/PyBaMM/pull/1261)
            solutions[i].solve_time = solve_time

        # Restore old y0
        model.y0 = old_y0

        # Report times
        if len(solutions) == 1:
            pybamm.logger.info("Finish solving {} ({})".format(model.name, termination))
            pybamm.logger.info(
                (
                    "Set-up time: {}, Solve time: {} (of which integration time: {}), "
                    "Total time: {}"
                ).format(
                    solutions[0].set_up_time,
                    solutions[0].solve_time,
                    solutions[0].integration_time,
                    solutions[0].total_time,
                )
            )
        else:
            pybamm.logger.info("Finish solving {} for all inputs".format(model.name))
            pybamm.logger.info(
                ("Set-up time: {}, Solve time: {}, Total time: {}").format(
                    solutions[0].set_up_time,
                    solutions[0].solve_time,
                    solutions[0].total_time,
                )
            )

        # Raise error if solutions[0] only contains one timestep (except for algebraic
        # solvers, where we may only expect one time in the solution)
        if (
            self.algebraic_solver is False
            and len(solution.all_ts) == 1
            and len(solution.all_ts[0]) == 1
        ):
            raise pybamm.SolverError(
                "Solution time vector has length 1. "
                "Check whether simulation terminated too early."
            )

        # Return solution(s)
        if ninputs == 1:
            return solutions[0]
        else:
            return solutions

    def step(
        self,
        old_solution,
        model,
        dt,
        npts=2,
        external_variables=None,
        inputs=None,
        save=True,
    ):
        """
        Step the solution of the model forward by a given time increment. The
        first time this method is called it executes the necessary setup by
        calling `self.set_up(model)`.

        Parameters
        ----------
        old_solution : :class:`pybamm.Solution` or None
            The previous solution to be added to. If `None`, a new solution is created.
        model : :class:`pybamm.BaseModel`
            The model whose solution to calculate. Must have attributes rhs and
            initial_conditions
        dt : numeric type
            The timestep (in seconds) over which to step the solution
        npts : int, optional
            The number of points at which the solution will be returned during
            the step dt. default is 2 (returns the solution at t0 and t0 + dt).
        external_variables : dict
            A dictionary of external variables and their corresponding
            values at the current time
        inputs : dict, optional
            Any input parameters to pass to the model when solving
        save : bool
            Turn on to store the solution of all previous timesteps




        Raises
        ------
        :class:`pybamm.ModelError`
            If an empty model is passed (`model.rhs = {}` and `model.algebraic = {}` and
            `model.variables = {}`)

        """

        if old_solution is not None and not (
            old_solution.termination == "final time"
            or "[experiment]" in old_solution.termination
        ):
            # Return same solution as an event has already been triggered
            # With hack to allow stepping past experiment current / voltage cut-off
            return old_solution

        # Make sure model isn't empty
        if len(model.rhs) == 0 and len(model.algebraic) == 0:
            if not isinstance(self, pybamm.DummySolver):
                raise pybamm.ModelError(
                    "Cannot step empty model, use `pybamm.DummySolver` instead"
                )

        # Make sure dt is positive
        if dt <= 0:
            raise pybamm.SolverError("Step time must be positive")

        # Set timer
        timer = pybamm.Timer()

        # Set up external variables and inputs
        external_variables = external_variables or {}
        inputs = inputs or {}

        # Remove interpolant inputs as Casadi can't handle them
        if isinstance(inputs.get("Current input [A]"), pybamm.Interpolant):
            del inputs["Current input [A]"]
        elif isinstance(inputs.get("Voltage input [V]"), pybamm.Interpolant):
            del inputs["Voltage input [V]"]
        elif isinstance(inputs.get("Power input [W]"), pybamm.Interpolant):
            del inputs["Power input [W]"]
        ext_and_inputs = {**external_variables, **inputs}

        if old_solution is None:
            # Run set up on first step
            pybamm.logger.verbose(
                "Start stepping {} with {}".format(model.name, self.name)
            )

            self.set_up(model, ext_and_inputs)
            self.models_set_up.update(
                {model: {"initial conditions": model.concatenated_initial_conditions}}
            )
            t = 0.0
        elif model not in self.models_set_up:
            # Run set up if the model has changed

            self.set_up(model, ext_and_inputs)
            self.models_set_up.update(
                {model: {"initial conditions": model.concatenated_initial_conditions}}
            )

        if old_solution is not None:
            t = old_solution.all_ts[-1][-1]
            if old_solution.all_models[-1] == model:
                # initialize with old solution
                model.y0 = old_solution.all_ys[-1][:, -1]
            else:
                model.y0 = (
                    model.set_initial_conditions_from(old_solution)
                    .concatenated_initial_conditions.evaluate(0, inputs=ext_and_inputs)
                    .flatten()
                )
        set_up_time = timer.time()

        # (Re-)calculate consistent initial conditions
        self._set_initial_conditions(model, ext_and_inputs, update_rhs=False)

        # Non-dimensionalise dt
        dt_dimensionless = dt / model.timescale_eval

        # Step
        t_eval = np.linspace(t, t + dt_dimensionless, npts)
        pybamm.logger.verbose(
            "Stepping for {:.0f} < t < {:.0f}".format(
                t * model.timescale_eval,
                (t + dt_dimensionless) * model.timescale_eval,
            )
        )
        timer.reset()
        solution = self._integrate(model, t_eval, ext_and_inputs)
        solution.solve_time = timer.time()

        # Check if extrapolation occurred
        extrapolation = self.check_extrapolation(solution, model.events)
        if extrapolation:
            warnings.warn(
                "While solving {} extrapolation occurred for {}".format(
                    model.name, extrapolation
                ),
                pybamm.SolverWarning,
            )

        # Identify the event that caused termination and update the solution to
        # include the event time and state
        solution, termination = self.get_termination_reason(solution, model.events)

        # Assign setup time
        solution.set_up_time = set_up_time

        # Report times
        pybamm.logger.verbose("Finish stepping {} ({})".format(model.name, termination))
        pybamm.logger.verbose(
            (
                "Set-up time: {}, Step time: {} (of which integration time: {}), "
                "Total time: {}"
            ).format(
                solution.set_up_time,
                solution.solve_time,
                solution.integration_time,
                solution.total_time,
            )
        )

        # Return solution
        if save is False:
            return solution
        else:
            return old_solution + solution

    def get_termination_reason(self, solution, events):
        """
        Identify the cause for termination. In particular, if the solver terminated
        due to an event, (try to) pinpoint which event was responsible. If an event
        occurs the event time and state are added to the solution object.
        Note that the current approach (evaluating all the events and then finding which
        one is smallest at the final timestep) is pretty crude, but is the easiest one
        that works for all the different solvers.

        Parameters
        ----------
        solution : :class:`pybamm.Solution`
            The solution object
        events : dict
            Dictionary of events
        """
        termination_events = [
            x for x in events if x.event_type == pybamm.EventType.TERMINATION
        ]
        if solution.termination == "final time":
            return (
                solution,
                "the solver successfully reached the end of the integration interval",
            )
        elif solution.termination == "event":
            pybamm.logger.debug("Start post-processing events")
            if solution.closest_event_idx is not None:
                solution.termination = (
                    f"event: {termination_events[solution.closest_event_idx].name}"
                )
            else:
                # Get final event value
                final_event_values = {}

                for event in termination_events:
                    final_event_values[event.name] = abs(
                        event.expression.evaluate(
                            solution.t_event,
                            solution.y_event,
                            inputs=solution.all_inputs[-1],
                        )
                    )
                termination_event = min(final_event_values, key=final_event_values.get)

                # Check that it's actually an event
                if abs(final_event_values[termination_event]) > 0.1:  # pragma: no cover
                    # Hard to test this
                    raise pybamm.SolverError(
                        "Could not determine which event was triggered "
                        "(possibly due to NaNs)"
                    )
                # Add the event to the solution object
                solution.termination = "event: {}".format(termination_event)
            # Update t, y and inputs to include event time and state
            # Note: if the final entry of t is equal to the event time we skip
            # this (having duplicate entries causes an error later in ProcessedVariable)
            if solution.t_event != solution.all_ts[-1][-1]:
                event_sol = pybamm.Solution(
                    solution.t_event,
                    solution.y_event,
                    solution.all_models[-1],
                    solution.all_inputs[-1],
                    solution.t_event,
                    solution.y_event,
                    solution.termination,
                )
                event_sol.solve_time = 0
                event_sol.integration_time = 0
                solution = solution + event_sol

            pybamm.logger.debug("Finish post-processing events")
            return solution, solution.termination
        elif solution.termination == "success":
            return solution, solution.termination

    def check_extrapolation(self, solution, events):
        """
        Check if extrapolation occurred for any of the interpolants. Note that with the
        current approach (evaluating all the events at the solution times) some
        extrapolations might not be found if they only occurred for a small period of
        time.

        Parameters
        ----------
        solution : :class:`pybamm.Solution`
            The solution object
        events : dict
            Dictionary of events
        """
        extrap_events = {}

        for event in events:
            if event.event_type == pybamm.EventType.INTERPOLANT_EXTRAPOLATION:
                # First set to False, then loop through and change to True if any
                # events extrapolate
                extrap_events[event.name] = False
                # This might be a little bit slow but is ok for now
                for ts, ys, inputs in zip(
                    solution.all_ts, solution.all_ys, solution.all_inputs
                ):
                    for inner_idx, t in enumerate(ts):
                        y = ys[:, inner_idx]
                        if isinstance(y, casadi.DM):
                            y = y.full()
                        if (
                            event.expression.evaluate(t, y, inputs=inputs)
                            < self.extrap_tol
                        ):
                            extrap_events[event.name] = True

        # Add the event dictionaryto the solution object
        solution.extrap_events = extrap_events

        return [k for k, v in extrap_events.items() if v]

    def _set_up_ext_and_inputs(
        self, model, external_variables, inputs, calculate_sensitivities
    ):
        """Set up external variables and input parameters"""
        inputs = inputs or {}

        # Go through all input parameters that can be found in the model
        # If any of them are *not* provided by "inputs", a symbolic input parameter is
        # created, with appropriate size
        for input_param in model.input_parameters:
            name = input_param.name
            if name not in inputs:
                # Only allow symbolic inputs for CasadiSolver and CasadiAlgebraicSolver
                if not isinstance(
                    self, (pybamm.CasadiSolver, pybamm.CasadiAlgebraicSolver)
                ):
                    raise pybamm.SolverError(
                        "Only CasadiSolver and CasadiAlgebraicSolver "
                        "can have symbolic inputs"
                    )
                inputs[name] = casadi.MX.sym(name, input_param._expected_size)

        external_variables = external_variables or {}

        ordered_inputs_names = list(inputs.keys())
        ordered_inputs_names.sort()
        ordered_inputs = {name: inputs[name] for name in ordered_inputs_names}

        ext_and_inputs = {**external_variables, **ordered_inputs}
        return ext_and_inputs


class SolverCallable:
    """A class that will be called by the solver when integrating"""

    def __init__(self, function, name, model):
        self._function = function
        if isinstance(function, casadi.Function):
            self.form = "casadi"
        else:
            self.form = "python"
        self.name = name
        self.model = model
        self.timescale = self.model.timescale_eval

    def __call__(self, t, y, inputs):
        pybamm.logger.debug(
            "Evaluating {} for {} at t={}".format(
                self.name, self.model.name, t * self.timescale
            )
        )
        if self.name in ["RHS", "algebraic", "residuals", "event"]:

            return self.function(t, y, inputs).flatten()
        else:
            return self.function(t, y, inputs)

    def function(self, t, y, inputs):
        if self.form == "casadi":
            states_eval = self._function(t, y, inputs)
            if self.name in ["RHS", "algebraic", "residuals", "event"]:
                return states_eval.full()
            else:
                # keep jacobians sparse
                return states_eval
        else:
            return self._function(t, y, inputs=inputs, known_evals={})[0]


class SensitivityCallable:
    """A class that will be called by the solver when integrating"""

    def __init__(self, function, name, model, form):
        self._function = function
        self.form = form
        self.name = name
        self.model = model
        self.timescale = self.model.timescale_eval

    def __call__(self, t, y, inputs):
        pybamm.logger.debug(
            "Evaluating sensitivities of {} for {} at t={}".format(
                self.name, self.model.name, t * self.timescale
            )
        )
        return self.function(t, y, inputs)

    def function(self, t, y, inputs):
        if self.form == "casadi":
            return self._function(t, y, inputs)
        elif self.form == "jax":
            return self._function(t, y, inputs=inputs)
        else:
            ret_with_known_evals = self._function(t, y, inputs=inputs, known_evals={})
            return {k: v[0] for k, v in ret_with_known_evals.items()}


class Residuals(SolverCallable):
    """Returns information about residuals at time t and state y"""

    def __init__(self, function, name, model):
        super().__init__(function, name, model)
        if model.mass_matrix is not None:
            self.mass_matrix = model.mass_matrix.entries

    def __call__(self, t, y, ydot, inputs):
        states_eval = super().__call__(t, y, inputs)
        return states_eval - self.mass_matrix @ ydot


class InitialConditions(SolverCallable):
    """Returns initial conditions given inputs"""

    def __init__(self, function, model):
        super().__init__(function, "initial conditions", model)

    def __call__(self, inputs):
        if self.form == "casadi":
            if isinstance(inputs, dict):
                inputs = casadi.vertcat(*[x for x in inputs.values()])
            return self._function(0, self.y_dummy, inputs)
        else:
            return self._function(0, self.y_dummy, inputs=inputs).flatten()
