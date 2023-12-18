import pybamm
import numpy as np
import logging
import jax
from jax import lax
from jax import numpy as jnp
from jax.interpreters import ad
from jax.interpreters import mlir
from jax.interpreters import batching
from jax.interpreters.mlir import custom_call
from jax.lib import xla_client
from jax.tree_util import tree_flatten

import importlib.util
import importlib

idaklu_spec = importlib.util.find_spec("pybamm.solvers.idaklu")
if idaklu_spec is not None:
    try:
        idaklu = importlib.util.module_from_spec(idaklu_spec)
        if idaklu_spec.loader:
            idaklu_spec.loader.exec_module(idaklu)
    except ImportError:  # pragma: no cover
        idaklu_spec = None


class IDAKLUJax:
    def get_var(self, f, varname):
        """Helper function to extract a single variable from the jaxified expression"""

        def f_isolated(*args, **kwargs):
            out = f(*args, **kwargs)
            index = self.jax_output_variables.index(varname)
            if out.ndim == 0:
                return out
            elif out.ndim == 1:
                return out[index]
            else:
                return out[:, index]

        return f_isolated

    def get_vars(self, f, varnames):
        """Helper function to extract multiple variables from the jaxified expression"""

        def f_isolated(*args, **kwargs):
            out = f(*args, **kwargs)
            index = np.array(
                [self.jax_output_variables.index(varname) for varname in varnames]
            )
            if out.ndim == 0:
                return out
            elif out.ndim == 1:
                return out[index]
            else:
                return out[:, index]

        return f_isolated

    def jax_value(self, *, f=None, t=None, inputs=None, output_variables=None):
        """Helper function to compute the gradient of a jaxified expression"""
        try:
            f = f if f else self.jaxify_f
            t = t if t else self.jax_t_eval
            inputs = inputs if inputs else self.jax_inputs
            output_variables = (
                output_variables if output_variables else self.jax_output_variables
            )
        except AttributeError:
            raise pybamm.SolverError("jaxify() must be called before jax_grad()")
        d = {}
        for outvar in self.jax_output_variables:
            d[outvar] = jax.vmap(
                self.get_var(f, outvar),
                in_axes=(0, None),
            )(t, inputs)
        return d

    def jax_grad(self, *, f=None, t=None, inputs=None, output_variables=None):
        """Helper function to compute the gradient of a jaxified expression"""
        try:
            f = f if f else self.jaxify_f
            t = t if t else self.jax_t_eval
            inputs = inputs if inputs else self.jax_inputs
            output_variables = (
                output_variables if output_variables else self.jax_output_variables
            )
        except AttributeError:
            raise pybamm.SolverError("jaxify() must be called before jax_grad()")
        d = {}
        for outvar in self.jax_output_variables:
            d[outvar] = jax.vmap(
                jax.grad(
                    self.get_var(f, outvar),
                    argnums=1,
                ),
                in_axes=(0, None),
            )(t, inputs)
        return d

    def _jaxify_solve(self, t, invar, *inputs_values):
        logging.info("jaxify_solve: ", type(t))
        # Reconstruct dictionary of inputs
        d = self.jax_inputs.copy()
        for ix, (k, v) in enumerate(self.jax_inputs.items()):
            d[k] = inputs_values[ix]
        # Solver
        logging.debug("Solver:")
        logging.debug("  t_eval: ", self.jax_t_eval)
        logging.debug("  t: ", t)
        logging.debug("  invar: ", invar)
        logging.debug("  inputs: ", dict(d))
        logging.debug("  calculate_sensitivities: ", invar is not None)
        sim = self.jax_solver.solve(
            self.jax_model,
            self.jax_t_eval,
            inputs=dict(d),
            calculate_sensitivities=invar is not None,
        )
        if invar:
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

    # TODO: Still working on this function (called from c++)
    def _jax_solve(
        self,
        t: float,
        in1: float,
        in2: float,
    ) -> np.ndarray:
        print(" py:jax_solve: ", type(t), type(in1), type(in2))
        print(" py:t = ", type(t), t)
        print(" py:in1 = ", type(in1), in1)
        print(" py:in2 = ", type(in2), in2)
        t = np.array(t)
        # Returns a jax array
        out = self._jaxify_solve(t, None, in1, in2)
        # Convert to numpy array
        out = np.array(out)
        self.keep_out = out  # Keep result referenced in memory
        # Returns a 2D array of size (len(t), len(output_variables))
        print(" py:out = ", type(self.keep_out), self.keep_out)
        return self.keep_out

    def _register_solve(self):
        """Register the solve method with the IDAKLU solver"""
        idaklu.add_python_callback(self._jax_solve)  # TODO: This causes python to hang on exit

    def __del__(self):
        self._deallocate()

    def _deallocate(self):
        print("Deallocate")
        idaklu.add_python_callback(None)

    def jaxify(
        self,
        model,
        t_eval,
        *,
        output_variables=None,
        inputs=None,
        calculate_sensitivities=True,
    ):
        """JAXify the model and solver"""

        self.jax_solver = self
        self.jax_model = model
        self.jax_t_eval = t_eval
        self.jax_output_variables = output_variables
        self.jax_inputs = inputs
        self._register_solve()

        # JAX PRIMITIVE DEFINITION

        f_p = jax.core.Primitive("f")
        f_p.multiple_results = False  # Returns a single multi-dimensional array

        def f(t, inputs):
            """
            Params:
                t : time
                inputs : dictionary of input values, e.g.
                         {'Current function [A]': 0.222, 'Separator porosity': 0.3}
            """
            logging.info("f: ", type(t), type(inputs))
            flatargs, treedef = tree_flatten((t, inputs))
            out = f_p.bind(*flatargs)
            logging.debug("f [exit]: ", (out))
            return out

        self.jaxify_f = f

        @f_p.def_impl
        def f_impl(t, *inputs):
            """Concrete implementation of Primitive"""
            logging.info("f_impl")
            term_v = self._jaxify_solve(t, None, *inputs)
            logging.debug("f_impl [exit]: ", (type(term_v), term_v))
            return term_v

        @f_p.def_abstract_eval
        def f_abstract_eval(t, *inputs):
            """Abstract evaluation of Primitive
            Takes abstractions of inputs, returned ShapedArray for result of primitive
            """
            logging.info("f_abstract_eval")
            if f_p.multiple_results:
                shape = t.shape
                dtype = jax.dtypes.canonicalize_dtype(t.dtype)
                return (jax.core.ShapedArray(shape, dtype),) * len(output_variables)
            else:
                shape = t.shape
                dtype = jax.dtypes.canonicalize_dtype(t.dtype)
                # y_aval = jax.core.ShapedArray(shape, t.dtype)
                y_aval = jax.core.ShapedArray((*t.shape, len(output_variables)), dtype)
                return y_aval

        def f_batch(args, batch_axes):
            """Batching rule for Primitive
            Takes batched inputs, returns batched outputs and batched axes
            """
            logging.info("f_batch: ", type(args), type(batch_axes))
            t = args[0]
            inputs = args[1:]
            if batch_axes[0] is not None and all([b is None for b in batch_axes[1:]]):
                # Temporal batching
                if t.ndim == 0:
                    return f_p.bind(t, *inputs), None
                return jnp.stack(list(map(lambda tp: f_p.bind(tp, *inputs), t))), 0
            else:
                raise NotImplementedError(
                    f"jaxify: batching not implemented for batch_axes = {batch_axes}"
                )

        batching.primitive_batchers[f_p] = f_batch

        # JVP / Forward-mode autodiff / J.v len(v)=num_inputs / len(return)=num_outputs

        def f_jvp(primals, tangents):
            logging.info("f_jvp: ", *list(map(type, (*primals, *tangents))))

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
            logging.debug("f_jvp [exit]: ", (type(y), y), (type(y_dot), y_dot))
            return y, y_dot

        ad.primitive_jvps[f_p] = f_jvp

        f_jvp_p = jax.core.Primitive("f_jvp")

        @f_jvp_p.def_impl
        def f_jvp_eval(*args):
            logging.info("f_jvp_p_eval: ", type(args))
            primals = args[: len(args) // 2]
            tangents = args[len(args) // 2 :]
            t = primals[0]
            inputs = primals[1:]
            inputs_t = tangents[1:]

            if t.ndim == 0:
                y_dot = jnp.zeros_like(t)
            else:
                # This permits direct vector indexing with time for jaxfwd
                y_dot = jnp.zeros((len(t), len(output_variables)))
            for index, value in enumerate(inputs_t):
                # Skipping zero values greatly improves performance
                if value > 0.0:
                    invar = list(self.jax_inputs.keys())[index]
                    js = self._jaxify_solve(t, invar, *inputs)
                    if js.ndim == 0:
                        js = jnp.array([js])
                    if js.ndim == 1 and t.ndim > 0:
                        # This permits direct vector indexing with time
                        js = js.reshape((t.shape[0], 1))
                    y_dot += value * js
            return y_dot

        def f_jvp_batch(args, batch_axes):
            logging.info("f_jvp_batch")
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
                if t.ndim == 0:
                    return f_jvp_p.bind(t, *inputs), None
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
                )

        batching.primitive_batchers[f_jvp_p] = f_jvp_batch

        @f_jvp_p.def_abstract_eval
        def f_jvp_abstract_eval(*args):
            logging.info("f_jvp_abstract_eval")
            primals = args[: len(args) // 2]
            t = primals[0]
            out = jax.core.ShapedArray((*t.shape, len(output_variables)), t.dtype)
            logging.info("<- f_jvp_abstract_eval")
            return out

        def f_jvp_transpose(y_bar, *args):
            # Note: y_bar indexes the OUTPUT variable, e.g. [1, 0, 0] is the
            # first of three outputs. The function returns primals and tangents
            # corresponding to how each of the inputs derives that output, e.g.
            #   (..., dout/din1, dout/din2)
            logging.info("f_jvp_transpose")
            primals = args[: len(args) // 2]

            tangents_out = []
            for invar in self.jax_inputs.keys():
                js = f_vjp_p.bind(y_bar, invar, *primals)
                tangents_out.append(js)

            out = (
                None,
                *([None] * len(tangents_out)),  # primals
                None,
                *tangents_out,  # tangents
            )
            logging.debug("<- f_jvp_transpose")
            return out

        ad.primitive_transposes[f_jvp_p] = f_jvp_transpose

        f_vjp_p = jax.core.Primitive("f_vjp")

        def f_vjp(y_bar, invar, *primals):
            logging.info("f_vjp")
            return f_vjp_p.bind(y_bar, invar, *primals)

        @f_vjp_p.def_impl
        def f_vjp_impl(y_bar, invar, *primals):
            logging.info("f_vjp_p_impl")
            t = primals[0]
            inputs = primals[1:]

            if t.ndim == 0:
                # scalar time input
                y_dot = jnp.zeros_like(t)
                js = self._jaxify_solve(t, invar, *inputs)
                if js.ndim == 0:
                    js = jnp.array([js])
                for index, value in enumerate(y_bar):
                    if value > 0.0:
                        y_dot += value * js[index]
            else:
                # vector time input
                js = self._jaxify_solve(t, invar, *inputs)
                if len(output_variables) == 1:
                    js = js.reshape((len(t), 1))
                y_dot = jnp.zeros(())
                for ix, y_outvar in enumerate(y_bar.T):
                    y_dot += jnp.dot(y_outvar, js[:, ix])
            logging.debug("<- f_vjp_p_impl")
            return y_dot

        def f_vjp_batch(args, batch_axes):
            logging.info("f_vjp_p_batch")
            y_bars, invar, t, *inputs = args

            if batch_axes[0] is not None and all([b is None for b in batch_axes[1:]]):
                # Batch over y_bar
                if y_bars.ndim <= 1:
                    return jnp.stack(f_vjp(*args)), 0
                out = list(map(lambda yb: f_vjp(yb, invar, t, *inputs), y_bars))
                return jnp.stack(out), 0
            elif (
                batch_axes[2] is not None
                and all([b is None for b in batch_axes[:2]])
                and all([b is None for b in batch_axes[3:]])
            ):
                # Batch over time
                if t.ndim == 0:
                    return f_vjp(*args), None
                out = list(map(lambda yt: f_vjp(y_bars, invar, yt, *inputs), t))
                return jnp.stack(out), 0
            else:
                raise Exception(
                    "Batch mode not supported for batch_axes = ", batch_axes
                )

        batching.primitive_batchers[f_vjp_p] = f_vjp_batch

        for _name, _value in idaklu.registrations().items():
            xla_client.register_custom_call_target(_name, _value, platform="cpu")

        def f_lowering_cpu(ctx, t, *inputs):
            t_aval = ctx.avals_in[0]
            np_dtype = np.dtype(t_aval.dtype)
            if np_dtype == np.float32:
                op_name = "cpu_idaklu_f32"
                op_dtype = mlir.ir.F32Type.get()
            elif np_dtype == np.float64:
                op_name = "cpu_idaklu_f64"
                op_dtype = mlir.ir.F64Type.get()
            else:
                raise NotImplementedError(f"Unsupported dtype {np_dtype}")

            dtype_t = mlir.ir.RankedTensorType(t.type)
            dims_t = dtype_t.shape
            layout_t = tuple(range(len(dims_t) - 1, -1, -1))
            size_t = np.prod(dims_t).astype(np.int64)

            y_aval = ctx.avals_out[0]
            dtype_out = mlir.ir.RankedTensorType.get(y_aval.shape, op_dtype)
            dims_out = dtype_out.shape
            layout_out = tuple(range(len(dims_out) - 1, -1, -1))

            input_aval = ctx.avals_in[1]
            dtype_input = mlir.ir.RankedTensorType.get(input_aval.shape, op_dtype)
            dims_input = dtype_input.shape
            layout_input = tuple(range(len(dims_input) - 1, -1, -1))

            results = custom_call(
                op_name,
                # Output types
                result_types=[dtype_out],
                # The inputs
                operands=[
                    mlir.ir_constant(size_t),  # 'size' argument
                    mlir.ir_constant(len(output_variables)),  # 'vars' argument
                    t,
                    *inputs,
                ],
                # Layout specification
                operand_layouts=[
                    (),  # 'size'
                    (),  # 'vars'
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

        return f
