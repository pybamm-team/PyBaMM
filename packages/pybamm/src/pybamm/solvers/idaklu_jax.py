from __future__ import annotations

import logging
import threading
import warnings
from functools import lru_cache

import numpy as np
import numpy.typing as npt
from pybammsolvers import idaklu

import pybamm

logger = logging.getLogger("pybamm.solvers.idaklu_jax")

if pybamm.has_jax():
    import jax

    try:
        from jax import ffi
    except ImportError:
        from jax.extend import ffi
    from jax import numpy as jnp
    from jax._src.interpreters.mlir import custom_call
    from jax.interpreters import ad, batching, mlir

    # Handle JAX version compatibility for Primitive location
    try:
        from jax.core import Primitive
    except ImportError:
        from jax.extend.core import Primitive


def _row_major_layout(shape) -> tuple[int, ...]:
    """Return the row-major (C-order) layout of an array with the given shape"""
    return tuple(range(len(shape) - 1, -1, -1))


class IDAKLUJax:
    """JAX wrapper for IDAKLU solver

    Objects of this class should be created via an IDAKLUSolver object. Values
    and derivatives are evaluated exactly at the requested times. Times outside
    the solution, and derivatives with respect to time, evaluate to NaN.

    Log information is available for this module via the named
    'pybamm.solvers.idaklu_jax' logger.

    Parameters
    ----------
    solver : :class:`pybamm.IDAKLUSolver`
        The IDAKLU solver object to be wrapped
    """

    def __init__(
        self,
        solver,
        model,
        t_eval,
        output_variables=None,
        calculate_sensitivities=True,
        t_interp=None,
    ):
        if not pybamm.has_jax():
            pybamm.raise_jax_not_found()  # pragma: no cover
        self.jaxpr = (
            None  # JAX expression representing the IDAKLU-wrapped solver object
        )
        self.idaklu_jax_obj = None  # Low-level IDAKLU-JAX primitives object
        self.solver = solver  # Originating IDAKLU Solver object
        self.jax_inputs = {k: [] for k, v in model.get_parameter_info().items()}

        # JAXify the solver ready for use
        self.jaxify(
            model,
            t_eval,
            output_variables=output_variables,
            calculate_sensitivities=calculate_sensitivities,
            t_interp=t_interp,
        )

    def get_jaxpr(self):
        """Returns a JAX expression representing the IDAKLU-wrapped solver object

        Returns
        -------
        Callable
            A JAX expression with the following call signature:
                f(t, inputs=None)
            where:
                t : float | np.ndarray
                    Time sample or vector of time samples
                inputs : dict, optional
                    dictionary of input values, e.g.
                    {'Current function [A]': 0.222, 'Separator porosity': 0.3}
        """
        if self.jaxpr is None:
            raise pybamm.SolverError("jaxify() must be called before get_jaxpr()")
        return self.jaxpr

    def get_var(
        self,
        *args,
    ):
        """Helper function to extract a single variable

        Isolates a single variable from the model output.
        Can be called on a JAX expression (which returns a JAX expression), or on a
        numeric (np.ndarray) object (which returns a slice of the output).

        Example call using default JAX expression, returns a JAX expression::

            f = idaklu_jax.get_var("Voltage [V]")
            data = f(t, inputs=None)

        Example call using a custom function, returns a JAX expression::

            f = idaklu_jax.get_var(jax.jit(f), "Voltage [V]")
            data = f(t, inputs=None)

        Example call to slice a matrix, returns an np.array::

            data = idaklu_jax.get_var(
                jax.fwd(f, argnums=1)(t_eval, inputs)['Current function [A]'],
                'Voltage [V]'
            )

        Parameters
        ----------
        f : Callable | np.ndarray, optional
            Expression or array from which to extract the target variable
        varname : str
            The name of the variable to extract

        Returns
        -------
        Callable
            If called with a JAX expression, returns a JAX expression with the following
            call signature:

                f(t, inputs=None)

            where:
                t : float | np.ndarray
                    Time sample or vector of time samples
                inputs : dict, optional
                    dictionary of input values, e.g.
                    {'Current function [A]': 0.222, 'Separator porosity': 0.3}
        np.ndarray
            If called with a numeric (np.ndarray) object, returns a slice of the output
            corresponding to the target variable.
        """

        # Identify the call signature
        if len(args) == 1:
            # Called on the default JAX expression
            f = self.jaxpr
            varname = args[0]
        elif len(args) == 2:
            # Called on a custom function
            f = args[0]
            varname = args[1]
        else:
            raise ValueError("Invalid call signature")

        # Utility function to slice the output
        def slice_out(out):
            index = self.jax_output_variables.index(varname)
            if out.ndim == 0:
                return out  # pragma: no cover
            return out[..., index]

        # If the jaxified expression is not a function, return a slice
        if not callable(f):
            return slice_out(f)

        # Otherwise, return a function that slices the output
        def f_isolated(*args, **kwargs):
            return slice_out(f(*args, **kwargs))

        return f_isolated

    def get_vars(
        self,
        *args,
    ):
        """Helper function to extract a list of variables

        Isolates a list of variables from the model output.
        Can be called on a JAX expression (which returns a JAX expression), or on a
        numeric (np.ndarray) object (which returns a slice of the output).

        Example call using default JAX expression, returns a JAX expression::

            f = idaklu_jax.get_vars(["Voltage [V]", "Current [A]"])
            data = f(t, inputs=None)

        Example call using a custom function, returns a JAX expression::

            f = idaklu_jax.get_vars(jax.jit(f), ["Voltage [V]", "Current [A]"])
            data = f(t, inputs=None)

        Example call to slice a matrix, returns an np.array::

            data = idaklu_jax.get_vars(
                jax.fwd(f, argnums=1)(t_eval, inputs)['Current function [A]'],
                ["Voltage [V]", "Current [A]"]
            )

        Parameters
        ----------
        f : Callable | np.ndarray, optional
            Expression or array from which to extract the target variables
        varname : list of str
            The names of the variables to extract

        Returns
        -------
        Callable
            If called with a JAX expression, returns a JAX expression with the following
            call signature:

                f(t, inputs=None)

            where:
                t : float | np.ndarray
                    Time sample or vector of time samples
                inputs : dict, optional
                    dictionary of input values, e.g.
                    {'Current function [A]': 0.222, 'Separator porosity': 0.3}
        np.ndarray
            If called with a numeric (np.ndarray) object, returns a slice of the output
            corresponding to the target variables.
        """

        # Identify the call signature
        if len(args) == 1:
            # Called on the default JAX expression
            f = self.jaxpr
            varnames = args[0]
        elif len(args) == 2:
            # Called on a custom function
            f = args[0]
            varnames = args[1]
        else:
            raise ValueError("Invalid call signature")

        # Utility function to slice the output
        def slice_out(out):
            index = np.array(
                [self.jax_output_variables.index(varname) for varname in varnames]
            )
            if out.ndim == 0:
                return out  # pragma: no cover
            return out[..., index]

        # If the jaxified expression is not a function, return a slice
        if not callable(f):
            return slice_out(f)

        # Otherwise, return a function that slices the output
        def f_isolated(*args, **kwargs):
            return slice_out(f(*args, **kwargs))

        return f_isolated

    def jax_value(
        self,
        t: npt.NDArray[np.float64] | None = None,
        inputs: dict | None = None,
        output_variables: list[str] | None = None,
    ):
        """Helper function to evaluate a jaxified expression

        Returns a numeric (np.ndarray) object (not a JAX expression).
        Parameters are inferred from the base object, but can be overridden.

        Parameters
        ----------
        t : float | np.ndarray
            Time sample or vector of time samples
        inputs : dict
            dictionary of input values
        output_variables : list of str, optional
            The variables to be returned. If None, the variables in the model are used.
        """
        if self.jaxpr is None:
            raise pybamm.SolverError("jaxify() must be called before get_jaxpr()")
        output_variables = (
            output_variables if output_variables else self.jax_output_variables
        )
        d = {}
        for outvar in output_variables:
            d[outvar] = jax.vmap(
                self.get_var(outvar),
                in_axes=(0, None),
            )(t, inputs)
        return d

    def jax_grad(
        self,
        t: npt.NDArray[np.float64] | None = None,
        inputs: dict | None = None,
        output_variables: list[str] | None = None,
    ):
        """Helper function to compute the gradient of a jaxified expression

        Returns a numeric (np.ndarray) object (not a JAX expression).
        Parameters are inferred from the base object, but can be overridden.

        Parameters
        ----------
        t : float | np.ndarray
            Time sample or vector of time samples
        inputs : dict
            dictionary of input values
        output_variables : list of str, optional
            The variables to be returned. If None, the variables in the model are used.
        """
        if self.jaxpr is None:
            raise pybamm.SolverError("jaxify() must be called before get_jaxpr()")
        output_variables = (
            output_variables if output_variables else self.jax_output_variables
        )
        d = {}
        for outvar in output_variables:
            d[outvar] = jax.vmap(
                jax.grad(
                    self.get_var(outvar),
                    argnums=1,
                ),
                in_axes=(0, None),
            )(t, inputs)
        return d

    class _hashabledict(dict):
        def __hash__(self):
            return hash(tuple(sorted(self.items())))

    @lru_cache(maxsize=1)  # noqa: B019
    def _cached_solve(solver, model, t_hashable, *args, **kwargs):
        """Cache the last solve for reuse

        Uses a threading lock to prevent concurrent access to the solver,
        which is not thread-safe during setup and solve operations.

        Note: This is a static method cached at the class level, where
        'solver' is the IDAKLUSolver instance.
        """
        # Get or create a lock for this solver instance
        if not hasattr(solver, "_idaklu_jax_lock"):
            solver._idaklu_jax_lock = threading.Lock()

        with solver._idaklu_jax_lock:
            return solver.solve(model, t_hashable, *args, **kwargs)

    def _jaxify_solve(self, t, invar, *inputs_values):
        """Solve the model and evaluate the outputs or their sensitivities at ``t``

        The requested times are added to the solver's interpolation times, so the
        values and sensitivities are both exact at ``t``. The last solve is cached.

        Parameters
        ----------
        t : float | np.ndarray
            Time sample(s), of any shape
        invar : str | int | None
            Name or index of the input to differentiate with respect to, or None
            to return the output values
        *inputs_values : float
            Input values, ordered as ``self.jax_input_names``

        Returns
        -------
        np.ndarray
            Array of shape ``(*t.shape, len(self.jax_output_variables))``, which is
            NaN at times outside the solution
        """
        t = np.asarray(t, dtype=np.float64)
        t_flat = t.reshape(-1)
        inputs = self._hashabledict(
            zip(
                self.jax_input_names,
                (np.asarray(value, dtype=np.float64).item() for value in inputs_values),
                strict=True,
            )
        )
        t_eval = np.asarray(self.jax_t_eval, dtype=np.float64)
        in_range = t_flat[(t_flat >= t_eval[0]) & (t_flat <= t_eval[-1])]
        t_interp = np.union1d(np.asarray(self.jax_t_interp, dtype=np.float64), in_range)
        logger.debug("_jaxify_solve:")
        logger.debug(f"  t_eval: {self.jax_t_eval}")
        logger.debug(f"  t_interp: {t_interp}")
        logger.debug(f"  t: {t}")
        logger.debug(f"  invar: {invar}")
        logger.debug(f"  inputs: {dict(inputs)}")
        sim = IDAKLUJax._cached_solve(
            self.solver,
            self.jax_model,
            tuple(self.jax_t_eval),
            inputs=inputs,
            calculate_sensitivities=self.jax_calculate_sensitivities,
            t_interp=tuple(t_interp),
        )
        # The solver stores every interpolation time exactly, so match rows exactly
        rows = np.minimum(np.searchsorted(sim.t, t_flat), sim.t.size - 1)
        found = sim.t[rows] == t_flat
        if invar is None:
            columns = [sim[outvar].data for outvar in self.jax_output_variables]
        else:
            if not isinstance(invar, str):
                invar = self.jax_input_names[int(invar)]
            columns = [
                sim[outvar].sensitivities[invar] for outvar in self.jax_output_variables
            ]
        out = np.stack([np.asarray(c).reshape(-1)[rows] for c in columns], axis=-1)
        out[~found] = np.nan
        return out.reshape(*t.shape, len(self.jax_output_variables))

    def _jax_solve_array_inputs(self, t, inputs_array):
        """Wrapper for _jax_solve used by IDAKLU callback

        This version assumes all parameters are provided as np.ndarray vectors
        """
        logger.info("jax_solve_array_inputs")
        logger.debug(f"  t: {type(t)}, {t}")
        logger.debug(f"  inputs_array: {type(inputs_array)}, {inputs_array}")
        inputs = tuple([k for k in inputs_array])
        logger.debug(f"  inputs: {type(inputs)}, {inputs}")
        return self._jax_solve(t, *inputs)

    def _jax_solve(
        self,
        t: float | npt.NDArray[np.float64],
        *inputs,
    ) -> npt.NDArray[np.float64]:
        """Solver implementation used by f-bind"""
        logger.info("jax_solve")
        logger.debug(f"  t: {type(t)}, {t}")
        logger.debug(f"  inputs: {type(inputs)}, {inputs}")
        return self._jaxify_solve(t, None, *inputs)

    def _jax_jvp_impl(
        self,
        *args: npt.NDArray[np.float64],
    ):
        """JVP implementation used by f_jvp bind"""
        primals = args[: len(args) // 2]
        tangents = args[len(args) // 2 :]
        t = np.asarray(primals[0], dtype=np.float64)
        inputs = primals[1:]
        t_dot = np.asarray(tangents[0])
        inputs_dot = tangents[1:]

        y_dot = np.zeros((*t.shape, len(self.jax_output_variables)))
        for invar, value in zip(self.jax_input_names, inputs_dot, strict=True):
            value = np.asarray(value, dtype=np.float64)
            # Skipping exact zeros avoids needless sensitivity evaluations
            if not np.any(value):
                continue
            y_dot += value * self._jaxify_solve(t, invar, *inputs)
        # Time derivatives are unavailable, so flag them rather than return zero
        y_dot[np.broadcast_to(t_dot != 0, t.shape)] = np.nan
        return y_dot

    def _jax_jvp_impl_array_inputs(
        self,
        primal_t,
        primal_inputs,
        tangent_t,
        tangent_inputs,
    ):
        """Wrapper for JVP implementation used by IDAKLU callback

        This version assumes all parameters are provided as np.ndarray vectors
        """
        primals = primal_t, *tuple([k for k in primal_inputs])
        tangents = tangent_t, *tuple([k for k in tangent_inputs])
        return self._jax_jvp_impl(*primals, *tangents)

    def _jax_vjp_impl(
        self,
        y_bar: npt.NDArray[np.float64],
        invar: str | int,  # index or name of input variable
        *primals: npt.NDArray[np.float64],
    ):
        """VJP implementation used by f_vjp bind"""
        logger.info("py:f_vjp_p_impl")
        logger.debug(f"  py:y_bar: {type(y_bar)}, {y_bar}")
        logger.debug(f"  py:invar: {type(invar)}, {invar}")
        logger.debug(f"  py:primals: {type(primals)}, {primals}")

        t = primals[0]
        inputs = primals[1:]

        sensitivities = self._jaxify_solve(t, invar, *inputs)
        y_bar = np.reshape(np.asarray(y_bar, dtype=np.float64), sensitivities.shape)
        # Zero cotangents are skipped so that unavailable (NaN) times stay masked
        mask = y_bar != 0
        out = np.array(np.sum(y_bar[mask] * sensitivities[mask]))
        logger.debug(f"_jax_vjp_impl [exit]: {out}")
        return out

    def _jax_vjp_impl_array_inputs(
        self,
        y_bar,
        y_bar_s0,
        y_bar_s1,
        invar,
        primal_t,
        primal_inputs,
    ):
        """Wrapper for VJP implementation used by IDAKLU callback

        This version assumes all parameters are provided as np.ndarray vectors
        """
        # Reshape y_bar
        logger.debug(f"Reshaping y_bar to ({y_bar_s0}, {y_bar_s1})")
        y_bar = y_bar.reshape(y_bar_s0, y_bar_s1)
        logger.debug(f"y_bar is now: {y_bar}")
        primals = primal_t, *tuple([k for k in primal_inputs])
        return self._jax_vjp_impl(y_bar, invar, *primals)

    def _register_callbacks(self):
        """Register the solve method with the IDAKLU solver"""
        logger.info("_register_callbacks")
        self.idaklu_jax_obj.register_callbacks(
            self._jax_solve_array_inputs,
            self._jax_jvp_impl_array_inputs,
            self._jax_vjp_impl_array_inputs,
        )

    def _unique_name(self):
        """Return a unique name for this solver object for naming the JAX primitives"""
        return f"{self.idaklu_jax_obj.get_index()}"

    def jaxify(
        self,
        model,
        t_eval,
        *,
        output_variables=None,
        calculate_sensitivities=True,
        t_interp=None,
    ):
        """JAXify the model and solver

        Creates a JAX expression representing the IDAKLU-wrapped solver
        object.

        Parameters
        ----------
        model : :class:`pybamm.BaseModel`
            The model to be solved
        t_eval : numeric type, optional
            The times at which to stop the integration due to a discontinuity in time.
        output_variables : list of str, optional
            The variables to be returned. If None, the variables in the model are used.
        calculate_sensitivities : bool, optional
            Whether to calculate sensitivities. Default is True.
        t_interp : None, list or ndarray, optional
            Extra times (in seconds) to store in every solve, so that calls at those
            times share one cached solve. Defaults to None.
        """
        if self.jaxpr is not None:
            warnings.warn(
                "JAX expression has already been created. "
                "Overwriting with new expression.",
                UserWarning,
                stacklevel=2,
            )
        self.jaxpr = self._jaxify(
            model,
            t_eval,
            output_variables=output_variables,
            calculate_sensitivities=calculate_sensitivities,
            t_interp=t_interp,
        )
        return self.jaxpr

    def _jaxify(
        self,
        model,
        t_eval,
        *,
        output_variables=None,
        calculate_sensitivities=True,
        t_interp=None,
    ):
        """JAXify the model and solver"""

        self.jax_model = model
        # Compiled functions read these names at run time, so they must not depend
        # on the inputs a trace of f sees
        self.jax_input_names = sorted({p.name for p in model.input_parameters})
        self.jax_t_eval = t_eval
        if t_interp is None:
            t_interp = np.empty(0)
        self.jax_t_interp = t_interp
        self.jax_output_variables = (
            output_variables if output_variables else self.solver.output_variables
        )
        if not self.jax_output_variables:
            raise pybamm.SolverError("output_variables must be specified")
        self.jax_calculate_sensitivities = calculate_sensitivities

        self.idaklu_jax_obj = idaklu.create_idaklu_jax()  # Create IDAKLU-JAX object
        self._register_callbacks()  # Register python methods as callbacks in IDAKLU-JAX

        for _name, _value in idaklu.registrations().items():
            ffi.register_ffi_target(
                f"{_name}_{self._unique_name()}", _value, platform="cpu", api_version=0
            )

        # --- JAX PRIMITIVE DEFINITION ------------------------------------------------

        logger.debug(f"Creating new primitive: {self._unique_name()}")
        f_p = Primitive(f"f_{self._unique_name()}")
        f_p.multiple_results = False  # Returns a single multi-dimensional array

        def f(t, inputs=None):
            """Main function wrapper for the JAX primitive function

            Parameters
            ----------
                t : float | np.ndarray
                    Time sample or vector of time samples
                inputs : dict, optional
                    dictionary of input values, e.g.
                         {'Current function [A]': 0.222, 'Separator porosity': 0.3}
            """
            logger.info("f")
            inputs = {} if inputs is None else inputs
            missing = [name for name in self.jax_input_names if name not in inputs]
            if missing:
                raise pybamm.SolverError(f"No value provided for input: {missing}")
            # Look up by name, as dict flattening order depends on the dict type
            values = [inputs[name] for name in self.jax_input_names]
            if any(np.size(value) != 1 for value in values):
                raise pybamm.SolverError("IDAKLUJax only supports scalar inputs")
            # The solver reads each input as one float64, and JAX restores the
            # original shape and dtype of their cotangents
            values = [
                jnp.reshape(jnp.asarray(value, dtype=jnp.float64), ())
                for value in values
            ]
            self.jax_inputs = inputs
            out = f_p.bind(t, *values)
            return out

        self.jaxify_f = f

        @f_p.def_impl
        def f_impl(t, *inputs):
            """Concrete implementation of Primitive (used for non-jitted evaluation)"""
            logger.info("f_impl")
            term_v = jnp.asarray(self._jaxify_solve(t, None, *inputs))
            logger.debug(f"f_impl [exit]: {type(term_v)}, {term_v}")
            return term_v

        @f_p.def_abstract_eval
        def f_abstract_eval(t, *inputs):
            """Abstract evaluation of Primitive"""
            logger.info("f_abstract_eval")
            dtype = jax.dtypes.canonicalize_dtype(t.dtype)
            y_aval = jax.core.ShapedArray(
                (*t.shape, len(self.jax_output_variables)), dtype
            )
            return y_aval

        def f_batch(args, batch_axes):
            """Batch rule for Primitive

            Takes batched inputs, returns batched outputs and batched axes"""
            logger.info(f"f_batch: {type(args)}, {type(batch_axes)}")
            t = args[0]
            inputs = args[1:]
            if batch_axes[0] is not None and all(b is None for b in batch_axes[1:]):
                # Temporal batching: one bind over all times shares a single solve
                return f_p.bind(jnp.moveaxis(t, batch_axes[0], 0), *inputs), 0
            else:
                raise NotImplementedError(
                    f"jaxify: batching not implemented for batch_axes = {batch_axes}"
                )

        batching.primitive_batchers[f_p] = f_batch

        def f_lowering_cpu(ctx, t, *inputs):
            """CPU lowering rule for Primitive

            This function calls the IDAKLU-JAX custom call target, which reroutes the
            call to the python callbacks, which call the standard IDAKLU solver.
            """
            logger.info("f_lowering_cpu")

            t_aval = ctx.avals_in[0]
            np_dtype = np.dtype(t_aval.dtype)
            if np_dtype == np.float64:
                op_name = f"cpu_idaklu_f64_{self._unique_name()}"
                op_dtype = mlir.ir.F64Type.get()
            else:
                raise NotImplementedError(
                    f"Unsupported dtype {np_dtype}"
                )  # pragma: no cover

            dtype_t = mlir.ir.RankedTensorType(t.type)
            dims_t = dtype_t.shape
            layout_t = _row_major_layout(dims_t)
            size_t = np.prod(dims_t).astype(np.int64)

            layout_inputs = [_row_major_layout(aval.shape) for aval in ctx.avals_in[1:]]

            y_aval = ctx.avals_out[0]
            dtype_out = mlir.ir.RankedTensorType.get(y_aval.shape, op_dtype)
            dims_out = dtype_out.shape
            layout_out = _row_major_layout(dims_out)

            results = custom_call(
                op_name,
                # Output types
                result_types=[dtype_out],
                # The inputs
                operands=[
                    mlir.ir_constant(
                        np.int64(self.idaklu_jax_obj.get_index())
                    ),  # solver index reference
                    mlir.ir_constant(size_t),  # 'size' argument
                    mlir.ir_constant(
                        np.int64(len(self.jax_output_variables))
                    ),  # 'vars' argument
                    mlir.ir_constant(np.int64(len(inputs))),  # 'vars' argument
                    t,
                    *inputs,
                ],
                # Layout specification
                operand_layouts=[
                    (),  # solver index reference
                    (),  # 'size'
                    (),  # 'vars'
                    (),  # number of inputs
                    layout_t,  # t
                    *layout_inputs,  # inputs
                ],
                result_layouts=[layout_out],
            )
            return results.results

        mlir.register_lowering(
            f_p,
            f_lowering_cpu,
            platform="cpu",
        )

        # --- JAX PRIMITIVE JVP DEFINITION --------------------------------------------

        def f_jvp(primals, tangents):
            """Main wrapper for the JVP function"""
            logger.info("f_jvp")

            # Deal with Zero tangents
            def make_zero(prim, tan):
                return jnp.zeros_like(prim) if type(tan) is ad.Zero else tan

            zero_mapped_tangents = tuple(
                make_zero(pt[0], pt[1]) for pt in zip(primals, tangents, strict=False)
            )

            y = f_p.bind(*primals)
            y_dot = f_jvp_p.bind(
                *primals,
                *zero_mapped_tangents,
            )
            logger.debug(f"f_jvp [exit]: {type(y)}, {y}, {type(y_dot)}, {y_dot}")
            return y, y_dot

        ad.primitive_jvps[f_p] = f_jvp

        f_jvp_p = Primitive(f"f_jvp_{self._unique_name()}")

        @f_jvp_p.def_impl
        def f_jvp_eval(*args):
            """Concrete implementation of JVP primitive (for non-jitted evaluation)"""
            logger.info(f"f_jvp_p_eval: {type(args)}")
            return self._jax_jvp_impl(*args)

        def f_jvp_batch(args, batch_axes):
            """Batch rule for JVP primitive"""
            logger.info("f_jvp_batch")
            n_args = len(args) // 2
            t, *inputs = args[:n_args]
            t_dot, *inputs_dot = args[n_args:]
            t_axis, *inputs_axes = batch_axes[:n_args]
            t_dot_axis, *inputs_dot_axes = batch_axes[n_args:]

            if any(b is not None for b in inputs_axes):
                raise NotImplementedError(
                    "f_jvp_batch: batching not implemented for batch_axes = "
                    f"{batch_axes}"
                )  # pragma: no cover
            if t_axis is not None:
                t = jnp.moveaxis(t, t_axis, 0)
            time_mask = None
            if t_dot_axis is not None:
                # Time derivatives are unavailable, so they are flagged as NaN below
                time_mask = jnp.moveaxis(t_dot, t_dot_axis, 0) != 0
                t_dot = jnp.zeros_like(t)
            elif t_axis is not None:
                t_dot = jnp.broadcast_to(t_dot, t.shape)

            if all(b is None for b in inputs_dot_axes):
                # Temporal batching: one bind over all times shares a single solve
                y_dot = f_jvp_p.bind(t, *inputs, t_dot, *inputs_dot)
            else:
                # The JVP is linear in the tangents, so combine per-input sensitivities
                zeros = [jnp.zeros_like(x) for x in inputs]
                y_dot = f_jvp_p.bind(t, *inputs, t_dot, *zeros)
                if t_axis is None:
                    y_dot = y_dot[None]
                for index, (x_dot, axis) in enumerate(
                    zip(inputs_dot, inputs_dot_axes, strict=True)
                ):
                    one_hot = [
                        jnp.ones_like(x) if k == index else zero
                        for k, (x, zero) in enumerate(zip(inputs, zeros, strict=True))
                    ]
                    sensitivity = f_jvp_p.bind(t, *inputs, jnp.zeros_like(t), *one_hot)
                    if t_axis is None:
                        sensitivity = sensitivity[None]
                    if axis is not None:
                        x_dot = jnp.moveaxis(x_dot, axis, 0)
                        x_dot = x_dot.reshape(
                            x_dot.shape + (1,) * (sensitivity.ndim - x_dot.ndim)
                        )
                    y_dot = y_dot + x_dot * sensitivity
            if time_mask is not None:
                y_dot = jnp.where(time_mask[..., None], jnp.nan, y_dot)
            return y_dot, 0

        batching.primitive_batchers[f_jvp_p] = f_jvp_batch

        @f_jvp_p.def_abstract_eval
        def f_jvp_abstract_eval(*args):
            """Abstract evaluation of JVP primitive"""
            logger.info("f_jvp_abstract_eval")
            primals = args[: len(args) // 2]
            t = primals[0]
            out = jax.core.ShapedArray(
                (*t.shape, len(self.jax_output_variables)), t.dtype
            )
            logger.info("<- f_jvp_abstract_eval")
            return out

        def f_jvp_transpose(y_bar, *args):
            """Transpose rule for JVP primitive"""

            # Note: y_bar indexes the OUTPUT variable, e.g. [1, 0, 0] is the
            # first of three outputs. The function returns primals and tangents
            # corresponding to how each of the inputs derives that output, e.g.
            #   (..., dout/din1, dout/din2)
            logger.info("f_jvp_transpose")
            n_args = len(args) // 2
            primals = args[:n_args]
            t_dot, *inputs_dot = args[n_args:]

            t_bar = None
            if ad.is_undefined_primal(t_dot):
                # Time derivatives are unavailable, so flag them rather than return zero
                t_bar = jnp.where(jnp.any(y_bar != 0, axis=-1), jnp.nan, 0.0)
            # Tangent operands are ordered as self.jax_input_names
            inputs_bar = [
                f_vjp(y_bar, index, *primals) if ad.is_undefined_primal(x_dot) else None
                for index, x_dot in enumerate(inputs_dot)
            ]
            logger.debug("<- f_jvp_transpose")
            return (*([None] * n_args), t_bar, *inputs_bar)

        ad.primitive_transposes[f_jvp_p] = f_jvp_transpose

        def f_jvp_lowering_cpu(ctx, *args):
            """CPU lowering rule for JVP primitive"""
            logger.info("f_jvp_lowering_cpu")

            primals = args[: len(args) // 2]
            t_primal = primals[0]
            inputs_primals = primals[1:]

            tangents = args[len(args) // 2 :]
            t_tangent = tangents[0]
            inputs_tangents = tangents[1:]

            t_aval = ctx.avals_in[0]
            np_dtype = np.dtype(t_aval.dtype)
            if np_dtype == np.float64:
                op_name = f"cpu_idaklu_jvp_f64_{self._unique_name()}"
                op_dtype = mlir.ir.F64Type.get()
            else:
                raise NotImplementedError(
                    f"Unsupported dtype {np_dtype}"
                )  # pragma: no cover

            dtype_t = mlir.ir.RankedTensorType(t_primal.type)
            dims_t = dtype_t.shape
            layout_t_primal = _row_major_layout(dims_t)
            layout_t_tangent = layout_t_primal
            size_t = np.prod(dims_t).astype(np.int64)

            n_primals = len(primals)
            layout_inputs_primals = [
                _row_major_layout(aval.shape) for aval in ctx.avals_in[1:n_primals]
            ]
            layout_inputs_tangents = [
                _row_major_layout(aval.shape) for aval in ctx.avals_in[n_primals + 1 :]
            ]

            y_aval = ctx.avals_out[0]
            dtype_out = mlir.ir.RankedTensorType.get(y_aval.shape, op_dtype)
            dims_out = dtype_out.shape
            layout_out = _row_major_layout(dims_out)

            results = custom_call(
                op_name,
                # Output types
                result_types=[dtype_out],
                # The inputs
                operands=[
                    mlir.ir_constant(
                        np.int64(self.idaklu_jax_obj.get_index())
                    ),  # solver index reference
                    mlir.ir_constant(size_t),  # 'size' argument
                    mlir.ir_constant(
                        np.int64(len(self.jax_output_variables))
                    ),  # 'vars' argument
                    mlir.ir_constant(np.int64(len(inputs_primals))),  # 'vars' argument
                    t_primal,  # 't'
                    *inputs_primals,  # inputs
                    t_tangent,  # 't'
                    *inputs_tangents,  # inputs
                ],
                # Layout specification
                operand_layouts=[
                    (),  # solver index reference
                    (),  # 'size'
                    (),  # 'vars'
                    (),  # number of inputs
                    layout_t_primal,  # 't'
                    *layout_inputs_primals,  # inputs
                    layout_t_tangent,  # 't'
                    *layout_inputs_tangents,  # inputs
                ],
                result_layouts=[layout_out],
            )
            return results.results

        mlir.register_lowering(
            f_jvp_p,
            f_jvp_lowering_cpu,
            platform="cpu",
        )

        # --- JAX PRIMITIVE VJP DEFINITION --------------------------------------------

        f_vjp_p = Primitive(f"f_vjp_{self._unique_name()}")

        def f_vjp(y_bar, invar, *primals):
            """Main wrapper for the VJP function"""
            logger.info("f_vjp")
            logger.debug(f"  y_bar: {y_bar}, {type(y_bar)}, {y_bar.shape}")
            return f_vjp_p.bind(y_bar, invar, *primals)

        @f_vjp_p.def_impl
        def f_vjp_impl(y_bar, invar, *primals):
            """Concrete implementation of VJP primitive (for non-jitted evaluation)"""
            logger.info("f_vjp_impl")
            return self._jax_vjp_impl(y_bar, invar, *primals)

        @f_vjp_p.def_abstract_eval
        def f_vjp_abstract_eval(*args):
            """Abstract evaluation of VJP primitive"""
            logger.info("f_vjp_abstract_eval")
            y_bar = args[0]
            out = jax.core.ShapedArray((), y_bar.dtype)
            logger.debug("<- f_vjp_abstract_eval")
            return out

        def f_vjp_batch(args, batch_axes):
            """Batch rule for VJP primitive"""
            logger.info("f_vjp_p_batch")
            y_bar, invar, t, *inputs = args
            y_bar_axis, invar_axis, t_axis, *inputs_axes = batch_axes
            if invar_axis is not None or any(b is not None for b in inputs_axes):
                raise pybamm.SolverError(
                    f"Batch mode not supported for batch_axes = {batch_axes}"
                )  # pragma: no cover

            # One bind over all times returns the sensitivities from a single solve
            one_hot = [
                jnp.where(invar == index, jnp.ones_like(x), jnp.zeros_like(x))
                for index, x in enumerate(inputs)
            ]
            if t_axis is not None:
                t = jnp.moveaxis(t, t_axis, 0)
            sensitivities = f_jvp_p.bind(t, *inputs, jnp.zeros_like(t), *one_hot)
            if t_axis is None:
                sensitivities = sensitivities[None]
            if y_bar_axis is None:
                y_bar = y_bar[None]
            else:
                y_bar = jnp.moveaxis(y_bar, y_bar_axis, 0)
            # Zero cotangents are skipped so that unavailable (NaN) times stay masked
            product = jnp.where(y_bar != 0, y_bar * sensitivities, 0.0)
            return jnp.sum(product, axis=tuple(range(1, product.ndim))), 0

        batching.primitive_batchers[f_vjp_p] = f_vjp_batch

        def f_vjp_lowering_cpu(ctx, y_bar, invar, *primals):
            """CPU lowering rule for VJP primitive"""
            logger.info("f_vjp_lowering_cpu")

            t_primal = primals[0]
            inputs_primals = primals[1:]

            t_aval = ctx.avals_in[2]
            np_dtype = np.dtype(t_aval.dtype)
            if np_dtype == np.float64:
                op_name = f"cpu_idaklu_vjp_f64_{self._unique_name()}"
                op_dtype = mlir.ir.F64Type.get()
            else:
                raise NotImplementedError(
                    f"Unsupported dtype {np_dtype}"
                )  # pragma: no cover

            y_bar_aval = ctx.avals_in[0]
            dtype_y_bar = mlir.ir.RankedTensorType.get(y_bar_aval.shape, op_dtype)
            dims_y_bar = dtype_y_bar.shape
            logger.debug(f"  y_bar shape: {dims_y_bar}")
            layout_y_bar = _row_major_layout(dims_y_bar)
            # The callback takes y_bar as 2D, so collapse any leading time dimensions
            if len(dims_y_bar) > 1:
                y_bar_s0, y_bar_s1 = int(np.prod(dims_y_bar[:-1])), dims_y_bar[-1]
            else:
                y_bar_s0, y_bar_s1 = dims_y_bar[0], -1

            invar_aval = ctx.avals_in[1]
            dtype_invar = mlir.ir.RankedTensorType.get(invar_aval.shape, op_dtype)
            dims_invar = dtype_invar.shape
            layout_invar = _row_major_layout(dims_invar)

            dtype_t = mlir.ir.RankedTensorType(t_primal.type)
            dims_t = dtype_t.shape
            layout_t_primal = _row_major_layout(dims_t)
            size_t = np.prod(dims_t).astype(np.int64)

            layout_inputs_primals = [
                _row_major_layout(aval.shape) for aval in ctx.avals_in[3:]
            ]

            y_aval = ctx.avals_out[0]
            dtype_out = mlir.ir.RankedTensorType.get(y_aval.shape, op_dtype)
            dims_out = dtype_out.shape
            layout_out = _row_major_layout(dims_out)

            results = custom_call(
                op_name,
                # Output types
                result_types=[dtype_out],
                # The inputs
                operands=[
                    mlir.ir_constant(
                        np.int64(self.idaklu_jax_obj.get_index())
                    ),  # solver index reference
                    mlir.ir_constant(size_t),  # 'size' argument
                    mlir.ir_constant(np.int64(len(inputs_primals))),  # number of inputs
                    mlir.ir_constant(np.int64(y_bar_s0)),  # 'y_bar' shape[0]
                    mlir.ir_constant(np.int64(y_bar_s1)),  # 'y_bar' shape[1]
                    y_bar,  # 'y_bar'
                    invar,  # 'invar'
                    t_primal,  # 't'
                    *inputs_primals,  # inputs
                ],
                # Layout specification
                operand_layouts=[
                    (),  # solver index reference
                    (),  # 'size'
                    (),  # number of inputs
                    (),  # 'y_bar' shape[0]
                    (),  # 'y_bar' shape[1]
                    layout_y_bar,  # 'y_bar'
                    layout_invar,  # 'invar'
                    layout_t_primal,  # 't'
                    *layout_inputs_primals,  # inputs
                ],
                result_layouts=[layout_out],
            )
            return results.results

        mlir.register_lowering(
            f_vjp_p,
            f_vjp_lowering_cpu,
            platform="cpu",
        )

        return f
