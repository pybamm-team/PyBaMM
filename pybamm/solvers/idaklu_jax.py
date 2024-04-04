import pybamm
import numpy as np
import logging
import warnings
import numbers

from typing import Union

from functools import lru_cache

import importlib.util
import importlib

logger = logging.getLogger("pybamm.solvers.idaklu_jax")

idaklu_spec = importlib.util.find_spec("pybamm.solvers.idaklu")
if idaklu_spec is not None:
    try:
        idaklu = importlib.util.module_from_spec(idaklu_spec)
        if idaklu_spec.loader:
            idaklu_spec.loader.exec_module(idaklu)
    except ImportError:  # pragma: no cover
        idaklu_spec = None

if pybamm.have_jax():
    import jax
    from jax import lax
    from jax import numpy as jnp
    from jax.interpreters import ad
    from jax.interpreters import mlir
    from jax.interpreters import batching
    from jax.interpreters.mlir import custom_call
    from jax.lib import xla_client
    from jax.tree_util import tree_flatten


class IDAKLUJax:
    """JAX wrapper for IDAKLU solver

    Objects of this class should be created via an IDAKLUSolver object.

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
    ):
        if not pybamm.have_jax():
            raise ModuleNotFoundError(
                "Jax or jaxlib is not installed, please see https://docs.pybamm.org/en/latest/source/user_guide/installation/gnu-linux-mac.html#optional-jaxsolver"
            )  # pragma: no cover
        if not pybamm.have_idaklu():
            raise ModuleNotFoundError(
                "IDAKLU is not installed, please see https://docs.pybamm.org/en/latest/source/user_guide/installation/index.html"
            )  # pragma: no cover
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
            elif out.ndim == 1:
                return out[index]
            else:
                return out[:, index]

        # If the jaxified expression is not a function, return a slice
        if not callable(f):
            return slice_out(f)

        # Otherwise, return a function that slices the output
        def f_isolated(*args, **kwargs):
            return slice_out(self.jaxify_f(*args, **kwargs))

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
            elif out.ndim == 1:
                return out[index]
            else:
                return out[:, index]

        # If the jaxified expression is not a function, return a slice
        if not callable(f):
            return slice_out(f)

        # Otherwise, return a function that slices the output
        def f_isolated(*args, **kwargs):
            return slice_out(self.jaxify_f(*args, **kwargs))

        return f_isolated

    def jax_value(
        self,
        t: np.ndarray = None,
        inputs: Union[dict, None] = None,
        output_variables: Union[list[str], None] = None,
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
                self.get_var(outvar),
                in_axes=(0, None),
            )(t, inputs)
        return d

    def jax_grad(
        self,
        t: np.ndarray = None,
        inputs: Union[dict, None] = None,
        output_variables: Union[list[str], None] = None,
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
    def _cached_solve(self, model, t_hashable, *args, **kwargs):
        """Cache the last solve for reuse"""
        return self.solve(model, t_hashable, *args, **kwargs)

    def _jaxify_solve(self, t, invar, *inputs_values):
        """Solve the model using the IDAKLU solver

        This method is called by the JAX primitive definition and caches the last solve
        fo reuse.
        """
        # Reconstruct dictionary of inputs
        if self.jax_inputs is None:
            d = self._hashabledict()
        else:
            # Use hashable dictionaries for caching the solve
            d = self._hashabledict()
            for key, value in zip(self.jax_inputs.keys(), inputs_values):
                d[key] = value
        # Solver
        logger.debug("_jaxify_solve:")
        logger.debug(f"  t_eval: {self.jax_t_eval}")
        logger.debug(f"  t: {t}")
        logger.debug(f"  invar: {invar}")
        logger.debug(f"  inputs: {dict(d)}")
        logger.debug(f"  calculate_sensitivities: {invar is not None}")
        sim = IDAKLUJax._cached_solve(
            self.solver,
            self.jax_model,
            tuple(self.jax_t_eval),
            inputs=self._hashabledict(d),
            calculate_sensitivities=self.jax_calculate_sensitivities,
        )
        if invar is not None:
            if isinstance(invar, numbers.Number):
                invar = list(self.jax_inputs.keys())[invar]
            # Provide vector support for time
            if t.ndim == 0:
                t = np.array([t])
            tk = list(map(lambda t: np.argmin(abs(self.jax_t_eval - t)), t))
            out = jnp.array(
                [
                    jnp.array(sim[outvar].sensitivities[invar][tk])
                    for outvar in self.jax_output_variables
                ]
            ).squeeze()
            return out.T
        else:
            return jnp.array(
                [np.array(sim[outvar](t)) for outvar in self.jax_output_variables]
            ).T

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
        t: Union[float, np.ndarray],
        *inputs,
    ) -> np.ndarray:
        """Solver implementation used by f-bind"""
        logger.info("jax_solve")
        logger.debug(f"  t: {type(t)}, {t}")
        logger.debug(f"  inputs: {type(inputs)}, {inputs}")
        # Returns a jax array
        out = self._jaxify_solve(t, None, *inputs)
        # Convert to numpy array
        return np.array(out)

    def _jax_jvp_impl(
        self,
        *args: Union[np.ndarray],
    ):
        """JVP implementation used by f_jvp bind"""
        primals = args[: len(args) // 2]
        tangents = args[len(args) // 2 :]
        t = primals[0]
        inputs = primals[1:]
        inputs_t = tangents[1:]

        if t.ndim == 0:
            y_dot = jnp.zeros_like(t)
        else:
            # This permits direct vector indexing with time for jacfwd
            y_dot = jnp.zeros((len(t), len(self.jax_output_variables)))
        for index, value in enumerate(inputs_t):
            # Skipping zero values greatly improves performance
            if value > 0.0:
                invar = list(self.jax_inputs.keys())[index]
                js = self._jaxify_solve(t, invar, *inputs)
                if js.ndim == 0:
                    js = jnp.array([js])
                if js.ndim == 1 and t.ndim > 0:
                    # This permits direct vector indexing with time
                    js = js.reshape((t.shape[0], -1))
                y_dot += value * js

        return np.array(y_dot)

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
        y_bar: np.ndarray,
        invar: Union[str, int],  # index or name of input variable
        *primals: np.ndarray,
    ):
        """VJP implementation used by f_vjp bind"""
        logger.info("py:f_vjp_p_impl")
        logger.debug(f"  py:y_bar: {type(y_bar)}, {y_bar}")
        logger.debug(f"  py:invar: {type(invar)}, {invar}")
        logger.debug(f"  py:primals: {type(primals)}, {primals}")

        t = primals[0]
        inputs = primals[1:]

        if isinstance(invar, float):
            invar = round(invar)
        if isinstance(t, float):
            t = np.array(t)

        if t.ndim == 0 or (t.ndim == 1 and t.shape[0] == 1):
            # scalar time input
            logger.debug("scalar time")
            y_dot = jnp.zeros_like(t)
            js = self._jaxify_solve(t, invar, *inputs)
            if js.ndim == 0:
                js = jnp.array([js])
            for index, value in enumerate(y_bar):
                if value > 0.0:
                    y_dot += value * js[index]
        else:
            logger.debug("vector time")
            # vector time input
            js = self._jaxify_solve(t, invar, *inputs)
            if len(self.jax_output_variables) == 1 and len(t) > 1:
                js = np.array([js]).T
            y_dot = jnp.zeros(())
            for ix, y_outvar in enumerate(y_bar.T):
                y_dot += jnp.dot(y_outvar, js[:, ix])
        logger.debug(f"_jax_vjp_impl [exit]: {type(y_dot)}, {y_dot}, {y_dot.shape}")
        y_dot = np.array(y_dot)
        return y_dot

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
    ):
        """JAXify the model and solver

        Creates a JAX expression representing the IDAKLU-wrapped solver
        object.

        Parameters
        ----------
        model : :class:`pybamm.BaseModel`
            The model to be solved
        t_eval : numeric type, optional
            The times at which to compute the solution. If None, the times in the model
            are used.
        output_variables : list of str, optional
            The variables to be returned. If None, the variables in the model are used.
        calculate_sensitivities : bool, optional
            Whether to calculate sensitivities. Default is True.
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
        )
        return self.jaxpr

    def _jaxify(
        self,
        model,
        t_eval,
        *,
        output_variables=None,
        calculate_sensitivities=True,
    ):
        """JAXify the model and solver"""

        self.jax_model = model
        self.jax_t_eval = t_eval
        self.jax_output_variables = (
            output_variables if output_variables else self.solver.output_variables
        )
        if not self.jax_output_variables:
            raise pybamm.SolverError("output_variables must be specified")
        self.jax_calculate_sensitivities = calculate_sensitivities

        self.idaklu_jax_obj = idaklu.create_idaklu_jax()  # Create IDAKLU-JAX object
        self._register_callbacks()  # Register python methods as callbacks in IDAKLU-JAX

        for _name, _value in idaklu.registrations().items():
            xla_client.register_custom_call_target(
                f"{_name}_{self._unique_name()}", _value, platform="cpu"
            )

        # --- JAX PRIMITIVE DEFINITION ------------------------------------------------

        logger.debug(f"Creating new primitive: {self._unique_name()}")
        f_p = jax.core.Primitive(f"f_{self._unique_name()}")
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
            flatargs, treedef = tree_flatten((t, inputs))
            self.jax_inputs = inputs
            out = f_p.bind(*flatargs)
            return out

        self.jaxify_f = f

        @f_p.def_impl
        def f_impl(t, *inputs):
            """Concrete implementation of Primitive (used for non-jitted evaluation)"""
            logger.info("f_impl")
            term_v = self._jaxify_solve(t, None, *inputs)
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
            if batch_axes[0] is not None and all([b is None for b in batch_axes[1:]]):
                # Temporal batching
                return jnp.stack(list(map(lambda tp: f_p.bind(tp, *inputs), t))), 0
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
            layout_t = tuple(range(len(dims_t) - 1, -1, -1))
            size_t = np.prod(dims_t).astype(np.int64)

            input_aval = ctx.avals_in[1]
            dtype_input = mlir.ir.RankedTensorType.get(input_aval.shape, op_dtype)
            dims_input = dtype_input.shape
            layout_input = tuple(range(len(dims_input) - 1, -1, -1))

            y_aval = ctx.avals_out[0]
            dtype_out = mlir.ir.RankedTensorType.get(y_aval.shape, op_dtype)
            dims_out = dtype_out.shape
            layout_out = tuple(range(len(dims_out) - 1, -1, -1))

            results = custom_call(
                op_name,
                # Output types
                result_types=[dtype_out],
                # The inputs
                operands=[
                    mlir.ir_constant(
                        self.idaklu_jax_obj.get_index()
                    ),  # solver index reference
                    mlir.ir_constant(size_t),  # 'size' argument
                    mlir.ir_constant(len(self.jax_output_variables)),  # 'vars' argument
                    mlir.ir_constant(len(inputs)),  # 'vars' argument
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
                    *([layout_input] * len(inputs)),  # inputs
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
                return lax.zeros_like_array(prim) if type(tan) is ad.Zero else tan

            zero_mapped_tangents = tuple(
                map(lambda pt: make_zero(pt[0], pt[1]), zip(primals, tangents))
            )

            y = f_p.bind(*primals)
            y_dot = f_jvp_p.bind(
                *primals,
                *zero_mapped_tangents,
            )
            logger.debug(f"f_jvp [exit]: {type(y)}, {y}, {type(y_dot)}, {y_dot}")
            return y, y_dot

        ad.primitive_jvps[f_p] = f_jvp

        f_jvp_p = jax.core.Primitive(f"f_jvp_{self._unique_name()}")

        @f_jvp_p.def_impl
        def f_jvp_eval(*args):
            """Concrete implementation of JVP primitive (for non-jitted evaluation)"""
            logger.info(f"f_jvp_p_eval: {type(args)}")
            return self._jax_jvp_impl(*args)

        def f_jvp_batch(args, batch_axes):
            """Batch rule for JVP primitive"""
            logger.info("f_jvp_batch")
            primals = args[: len(args) // 2]
            tangents = args[len(args) // 2 :]
            batch_primals = batch_axes[: len(batch_axes) // 2]
            batch_tangents = batch_axes[len(batch_axes) // 2 :]

            if (
                batch_primals[0] is not None
                and all([b is None for b in batch_primals[1:]])
                and all([b is None for b in batch_tangents])
            ):
                # Temporal batching (primals) only
                t = primals[0]
                inputs = primals[1:]
                return (
                    jnp.stack(
                        list(map(lambda tp: f_jvp_p.bind(tp, *inputs, *tangents), t))
                    ),
                    0,
                )
            elif (
                batch_tangents[0] is not None
                and all([b is None for b in batch_tangents[1:]])
                and all([b is None for b in batch_primals])
            ):
                # Batch over derivates wrt time
                raise NotImplementedError(
                    "Taking the derivative with respect to time is not supported"
                )
            elif (
                batch_tangents[0] is None
                and any([b is not None for b in batch_tangents[1:]])
                and all([b is None for b in batch_primals])
            ):
                # Batch over (some combination of) inputs
                batch_axis_indices = [
                    i for i, b in enumerate(batch_tangents) if b is not None
                ]
                out = []
                for i in range(len(batch_axis_indices)):
                    tangents_item = list(tangents)
                    for k in range(len(batch_axis_indices)):
                        tangents_item[batch_axis_indices[k]] = tangents[
                            batch_axis_indices[k]
                        ][i]
                    out.append(f_jvp_p.bind(*primals, *tangents_item))
                return jnp.stack(out), 0
            else:
                raise NotImplementedError(
                    "f_jvp_batch: batching not implemented for batch_axes = "
                    f"{batch_axes}"
                )  # pragma: no cover

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
            primals = args[: len(args) // 2]

            tangents_out = []
            for invar in self.jax_inputs.keys():
                js = f_vjp(y_bar, invar, *primals)
                tangents_out.append(js)

            out = (
                None,
                *([None] * len(tangents_out)),  # primals
                None,
                *tangents_out,  # tangents
            )
            logger.debug("<- f_jvp_transpose")
            return out

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
            layout_t_primal = tuple(range(len(dims_t) - 1, -1, -1))
            layout_t_tangent = layout_t_primal
            size_t = np.prod(dims_t).astype(np.int64)

            input_aval = ctx.avals_in[1]
            dtype_input = mlir.ir.RankedTensorType.get(input_aval.shape, op_dtype)
            dims_input = dtype_input.shape
            layout_inputs_primals = tuple(range(len(dims_input) - 1, -1, -1))
            layout_inputs_tangents = layout_inputs_primals

            y_aval = ctx.avals_out[0]
            dtype_out = mlir.ir.RankedTensorType.get(y_aval.shape, op_dtype)
            dims_out = dtype_out.shape
            layout_out = tuple(range(len(dims_out) - 1, -1, -1))

            results = custom_call(
                op_name,
                # Output types
                result_types=[dtype_out],
                # The inputs
                operands=[
                    mlir.ir_constant(
                        self.idaklu_jax_obj.get_index()
                    ),  # solver index reference
                    mlir.ir_constant(size_t),  # 'size' argument
                    mlir.ir_constant(len(self.jax_output_variables)),  # 'vars' argument
                    mlir.ir_constant(len(inputs_primals)),  # 'vars' argument
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
                    *([layout_inputs_primals] * len(inputs_primals)),  # inputs
                    layout_t_tangent,  # 't'
                    *([layout_inputs_tangents] * len(inputs_tangents)),  # inputs
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

        f_vjp_p = jax.core.Primitive(f"f_vjp_{self._unique_name()}")

        def f_vjp(y_bar, invar, *primals):
            """Main wrapper for the VJP function"""
            logger.info("f_vjp")
            logger.debug(f"  y_bar: {y_bar}, {type(y_bar)}, {y_bar.shape}")
            if isinstance(invar, str):
                invar = list(self.jax_inputs.keys()).index(invar)
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
            primals = args[: len(args) // 2]
            t = primals[0]
            out = jax.core.ShapedArray((), t.dtype)
            logger.debug("<- f_vjp_abstract_eval")
            return out

        def f_vjp_batch(args, batch_axes):
            """Batch rule for VJP primitive"""
            logger.info("f_vjp_p_batch")
            y_bars, invar, t, *inputs = args

            if batch_axes[0] is not None and all([b is None for b in batch_axes[1:]]):
                # Batch over y_bar
                out = list(map(lambda yb: f_vjp(yb, invar, t, *inputs), y_bars))
                return jnp.stack(out), 0
            elif (
                batch_axes[2] is not None
                and all([b is None for b in batch_axes[:2]])
                and all([b is None for b in batch_axes[3:]])
            ):
                # Batch over time
                out = list(map(lambda yt: f_vjp(y_bars, invar, yt, *inputs), t))
                return jnp.stack(out), 0
            else:
                raise Exception(
                    "Batch mode not supported for batch_axes = ", batch_axes
                )  # pragma: no cover

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
            layout_y_bar = tuple(range(len(dims_y_bar) - 1, -1, -1))

            invar_aval = ctx.avals_in[1]
            dtype_invar = mlir.ir.RankedTensorType.get(invar_aval.shape, op_dtype)
            dims_invar = dtype_invar.shape
            layout_invar = tuple(range(len(dims_invar) - 1, -1, -1))

            dtype_t = mlir.ir.RankedTensorType(t_primal.type)
            dims_t = dtype_t.shape
            layout_t_primal = tuple(range(len(dims_t) - 1, -1, -1))
            size_t = np.prod(dims_t).astype(np.int64)

            input_aval = ctx.avals_in[3]
            dtype_input = mlir.ir.RankedTensorType.get(input_aval.shape, op_dtype)
            dims_input = dtype_input.shape
            layout_inputs_primals = tuple(range(len(dims_input) - 1, -1, -1))

            y_aval = ctx.avals_out[0]
            dtype_out = mlir.ir.RankedTensorType.get(y_aval.shape, op_dtype)
            dims_out = dtype_out.shape
            layout_out = tuple(range(len(dims_out) - 1, -1, -1))

            results = custom_call(
                op_name,
                # Output types
                result_types=[dtype_out],
                # The inputs
                operands=[
                    mlir.ir_constant(
                        self.idaklu_jax_obj.get_index()
                    ),  # solver index reference
                    mlir.ir_constant(size_t),  # 'size' argument
                    mlir.ir_constant(len(self.jax_inputs)),  # number of inputs
                    mlir.ir_constant(dims_y_bar[0]),  # 'y_bar' shape[0]
                    mlir.ir_constant(  # 'y_bar' shape[1]
                        dims_y_bar[1] if len(dims_y_bar) > 1 else -1
                    ),  # 'y_bar' argument
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
                    *([layout_inputs_primals] * len(inputs_primals)),  # inputs
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
