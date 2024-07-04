#
# Solver class using Scipy's adaptive time stepper
#
import numpy as onp

import pybamm

if pybamm.have_jax():
    import jax
    import jax.numpy as jnp
    from jax.experimental.ode import odeint


class JaxSolver(pybamm.BaseSolver):
    """
    Solve a discretised model using a JAX compiled solver.

    **Note**: this solver will not work with models that have
              termination events or are not converted to jax format

    Raises
    ------

    RuntimeError
        if model has any termination events

    RuntimeError
        if `model.convert_to_format != 'jax'`

    Parameters
    ----------
    method: str, optional (see `jax.experimental.ode.odeint` for details)
        * 'RK45' (default) uses jax.experimental.ode.odeint
        * 'BDF' uses custom jax_bdf_integrate (see `jax_bdf_integrate.py` for details)
    root_method: str, optional
        Method to use to calculate consistent initial conditions. By default this uses
        the newton chord method internal to the jax bdf solver, otherwise choose from
        the set of default options defined in docs for pybamm.BaseSolver
    rtol : float, optional
        The relative tolerance for the solver (default is 1e-6).
    atol : float, optional
        The absolute tolerance for the solver (default is 1e-6).
    extrap_tol : float, optional
        The tolerance to assert whether extrapolation occurs or not (default is 0).
    extra_options : dict, optional
        Any options to pass to the solver.
        Please consult `JAX documentation
        <https://github.com/google/jax/blob/master/jax/experimental/ode.py>`_
        for details.
    """

    def __init__(
        self,
        method="RK45",
        root_method=None,
        rtol=1e-6,
        atol=1e-6,
        extrap_tol=None,
        extra_options=None,
    ):
        if not pybamm.have_jax():
            raise ModuleNotFoundError(
                "Jax or jaxlib is not installed, please see https://docs.pybamm.org/en/latest/source/user_guide/installation/gnu-linux-mac.html#optional-jaxsolver"
            )

        # note: bdf solver itself calculates consistent initial conditions so can set
        # root_method to none, allow user to override this behavior
        super().__init__(
            method, rtol, atol, root_method=root_method, extrap_tol=extrap_tol
        )
        method_options = ["RK45", "BDF"]
        if method not in method_options:
            raise ValueError(f"method must be one of {method_options}")
        self.ode_solver = False
        if method == "RK45":
            self.ode_solver = True
        self.extra_options = extra_options or {}
        self.name = f"JAX solver ({method})"
        self._cached_solves = dict()
        pybamm.citations.register("jax2018")

    def get_solve(self, model, t_eval):
        """
        Return a compiled JAX function that solves an ode model with input arguments.

        Parameters
        ----------
        model : :class:`pybamm.BaseModel`
            The model whose solution to calculate.
        t_eval : :class:`numpy.array`, size (k,)
            The times at which to compute the solution

        Returns
        -------
        function
            A function with signature `f(inputs)`, where inputs are a dict containing
            any input parameters to pass to the model when solving

        """
        if model not in self._cached_solves:
            if model not in self._model_set_up:
                raise RuntimeError(
                    "Model is not set up for solving, run" "`solver.solve(model)` first"
                )

            self._cached_solves[model] = self.create_solve(model, t_eval)

        return self._cached_solves[model]

    def create_solve(self, model, t_eval):
        """
        Return a compiled JAX function that solves an ode model with input arguments.

        Parameters
        ----------
        model : :class:`pybamm.BaseModel`
            The model whose solution to calculate.
        t_eval : :class:`numpy.array`, size (k,)
            The times at which to compute the solution

        Returns
        -------
        function
            A function with signature `f(inputs)`, where inputs are a dict containing
            any input parameters to pass to the model when solving

        """
        if model.convert_to_format != "jax":
            raise RuntimeError(
                "Model must be converted to JAX to use this solver"
                " (i.e. `model.convert_to_format = 'jax')"
            )

        if model.terminate_events_eval:
            raise RuntimeError(
                "Terminate events not supported for this solver."
                " Model has the following events:"
                f" {model.events}.\nYou can remove events using `model.events = []`."
                " It might be useful to first solve the model using a"
                " different solver to obtain the time of the event, then"
                " re-solve using no events and a fixed"
                " end-time"
            )

        # Initial conditions, make sure they are an 0D array
        y0 = jnp.array(model.y0).reshape(-1)
        mass = None
        if self.method == "BDF":
            mass = model.mass_matrix.entries.toarray()
            print("stuff", mass.shape, model.len_rhs, model.len_alg)

        def stack_inputs(inputs: dict | list[dict]):
            if isinstance(inputs, dict):
                return jnp.array([x.reshape(-1, 1) for x in inputs.values()])
            if len(inputs) == 1:
                return jnp.array([x.reshape(-1, 1) for x in inputs[0].values()])
            arrays_to_stack = [
                jnp.array(x).reshape(-1, 1)
                for inputs in inputs
                for x in inputs.values()
            ]
            print("stacking", len(arrays_to_stack), arrays_to_stack[0].shape)
            return jnp.vstack(arrays_to_stack)

        def rhs_ode(y, t, inputs):
            return (model.rhs_eval(t, y, inputs),)

        def rhs_dae(y, t, inputs):
            return jnp.concatenate(
                [model.rhs_eval(t, y, inputs), model.algebraic_eval(t, y, inputs)]
            )

        def solve_model_rk45(inputs):
            y = odeint(
                rhs_ode,
                y0,
                t_eval,
                stack_inputs(inputs),
                rtol=self.rtol,
                atol=self.atol,
                **self.extra_options,
            )
            return jnp.transpose(y)

        def solve_model_bdf(inputs):
            y = pybamm.jax_bdf_integrate(
                rhs_dae,
                y0,
                t_eval,
                stack_inputs(inputs),
                rtol=self.rtol,
                atol=self.atol,
                mass=mass,
                **self.extra_options,
            )
            return jnp.transpose(y)

        if self.method == "RK45":
            return jax.jit(solve_model_rk45)
        else:
            return jax.jit(solve_model_bdf)

    def _integrate(self, model, t_eval, inputs=None):
        """
        Solve a model defined by dydt with initial conditions y0.

        Parameters
        ----------
        model : :class:`pybamm.BaseModel`
            The model whose solution to calculate.
        t_eval : :class:`numpy.array`, size (k,)
            The times at which to compute the solution
        inputs_list : list[dict], optional
            Any input parameters to pass to the model when solving

        Returns
        -------
        object
            An object containing the times and values of the solution, as well as
            various diagnostic messages.

        """
        inputs = inputs or [{}]

        timer = pybamm.Timer()
        if model not in self._cached_solves:
            self._cached_solves[model] = self.create_solve(model, t_eval)

        y = self._cached_solves[model](inputs)

        integration_time = timer.time()

        # convert to a normal numpy array
        y = onp.array(y)

        termination = "final time"

        sol = pybamm.Solution(
            t_eval,
            y,
            model,
            inputs[0],
            termination=termination,
            check_solution=False,
        )
        sol.integration_time = integration_time
        return sol.split(model.len_rhs, model.len_alg, inputs)
