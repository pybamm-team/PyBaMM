#
# Tests for the KLU-Jax interface class
#
from tests import TestCase
from jax.tree_util import tree_flatten
from parameterized import parameterized

import pybamm
import numpy as np
import jax
import jax.numpy as jnp
import unittest

inputs = {
    "Current function [A]": 0.222,
    "Separator porosity": 0.3,
}

model = pybamm.lithium_ion.DFN()
geometry = model.default_geometry
param = model.default_parameter_values
param.update({key: "[input]" for key in inputs.keys()})
param.process_geometry(geometry)
param.process_model(model)
var = pybamm.standard_spatial_vars
var_pts = {var.x_n: 20, var.x_s: 20, var.x_p: 20, var.r_n: 10, var.r_p: 10}
mesh = pybamm.Mesh(geometry, model.default_submesh_types, var_pts)
disc = pybamm.Discretisation(mesh, model.default_spatial_methods)
disc.process_model(model)
t_eval = np.linspace(0, 360, 10)
idaklu_solver = pybamm.IDAKLUSolver(rtol=1e-6, atol=1e-6)
solver = idaklu_solver

# Create surrogate data (using standard solver)
sim = idaklu_solver.solve(
    # model, t_eval=, inputs=, initial_conditions=, nproc=, calculate_sensitivities=
    model,
    t_eval,
    inputs=inputs,
    calculate_sensitivities=True,
)

# Get jax expression for IDAKLU solver
output_variables = [
    "Terminal voltage [V]",
    "Discharge capacity [A.h]",
    "Loss of lithium inventory [%]",
]
f1 = idaklu_solver.jaxify(
    model,
    t_eval,
    output_variables=output_variables[:1],
    inputs=inputs,
    calculate_sensitivities=True,
)
f3 = idaklu_solver.jaxify(
    model,
    t_eval,
    output_variables=output_variables,
    inputs=inputs,
    calculate_sensitivities=True,
)


# TEST

x = inputs
in_axes = (0, None)
k = 5

testcase = [
    (output_variables[:1], f1),  # single output
    (output_variables, f3),  # multiple outputs
]


@unittest.skipIf(
    not pybamm.have_idaklu() or not pybamm.have_jax(),
    "IDAKLU Solver and/or JAX are not installed",
)
class TestIDAKLUJax(TestCase):
    # Scalar evaluation

    @parameterized.expand(testcase)
    def test_f_scalar(self, output_variables, f):
        print("\nf (scalar):")
        out = f(t_eval[k], inputs)
        print(out)
        assert np.allclose(
            out, np.array([sim[outvar](t_eval[k]) for outvar in output_variables]).T
        )

    @parameterized.expand(testcase)
    def test_f_vector(self, output_variables, f):
        print("\nf (vector):")
        out = f(t_eval, inputs)
        print(out)
        assert np.allclose(
            out, np.array([sim[outvar](t_eval) for outvar in output_variables]).T
        )

    @parameterized.expand(testcase)
    def test_f_vmap(self, output_variables, f):
        print("\nf (vmap):")
        out = jax.vmap(f, in_axes=in_axes)(t_eval, x)
        print(out)
        assert np.allclose(
            out, np.array([sim[outvar](t_eval) for outvar in output_variables]).T
        )

    @parameterized.expand(testcase)
    def test_getvars_scalar(self, output_variables, f):
        # Get all vars (should mirror above outputs)
        print("\nget_vars (scalar)")
        out = idaklu_solver.get_vars(f, output_variables)(t_eval[k], x)
        print(out)
        assert np.allclose(
            out, np.array([sim[outvar](t_eval[k]) for outvar in output_variables]).T
        )

    @parameterized.expand(testcase)
    def test_getvars_vector(self, output_variables, f):
        print("\nget_vars (vector)")
        out = idaklu_solver.get_vars(f, output_variables)(t_eval, x)
        print(out)
        assert np.allclose(
            out, np.array([sim[outvar](t_eval) for outvar in output_variables]).T
        )

    @parameterized.expand(testcase)
    def test_getvars_vmap(self, output_variables, f):
        print("\nget_vars (vmap)")
        out = jax.vmap(
            idaklu_solver.get_vars(f, output_variables),
            in_axes=(0, None),
        )(t_eval, x)
        print(out)
        assert np.allclose(
            out, np.array([sim[outvar](t_eval) for outvar in output_variables]).T
        )

    @parameterized.expand(testcase)
    def test_getvar_scalar(self, output_variables, f):
        # Per variable checks
        for outvar in output_variables:
            print(f"\nget_var (scalar): {outvar}")
            out = idaklu_solver.get_var(f, outvar)(t_eval[k], x)
            print(out)
            assert np.allclose(out, sim[outvar](t_eval[k]))

    @parameterized.expand(testcase)
    def test_getvar_vector(self, output_variables, f):
        for outvar in output_variables:
            print(f"\nget_var (vector): {outvar}")
            out = idaklu_solver.get_var(f, outvar)(t_eval, x)
            print(out)
            assert np.allclose(out, sim[outvar](t_eval))

    @parameterized.expand(testcase)
    def test_getvar_vmap(self, output_variables, f):
        for outvar in output_variables:
            print(f"\nget_var (vmap): {outvar}")
            out = jax.vmap(
                idaklu_solver.get_var(f, outvar),
                in_axes=(0, None),
            )(t_eval, x)
            print(out)
            assert np.allclose(out, sim[outvar](t_eval))

    # Differentiation rules

    @parameterized.expand(testcase)
    def test_jacfwd_scalar(self, output_variables, f):
        print("\njac_fwd (scalar)")
        out = jax.jacfwd(f, argnums=1)(t_eval[k], x)
        print(out)
        flat_out, _ = tree_flatten(out)
        flat_out = np.array([f for f in flat_out]).flatten()
        check = np.array(
            [
                sim[outvar].sensitivities[invar][k]
                for invar in x
                for outvar in output_variables
            ]
        ).T
        assert np.allclose(flat_out, check.flatten())

    @parameterized.expand(testcase)
    def test_jacfwd_vector(self, output_variables, f):
        print("\njac_fwd (vector)")
        out = jax.jacfwd(f, argnums=1)(t_eval, x)
        print(out)
        flat_out, _ = tree_flatten(out)
        flat_out = np.concatenate(np.array([f for f in flat_out]), 1).T.flatten()
        check = np.array(
            [
                sim[outvar].sensitivities[invar]
                for invar in x
                for outvar in output_variables
            ]
        )
        assert np.allclose(
            flat_out, check.flatten()
        ), f"Got: {flat_out}\nExpected: {check}"

    @parameterized.expand(testcase)
    def test_jacfwd_vmap(self, output_variables, f):
        print("\njac_fwd (vmap)")
        out = jax.vmap(
            jax.jacfwd(f, argnums=1),
            in_axes=(0, None),
        )(t_eval, x)
        print(out)
        flat_out, _ = tree_flatten(out)
        flat_out = np.concatenate(np.array([f for f in flat_out]), 1).T.flatten()
        check = np.array(
            [
                sim[outvar].sensitivities[invar]
                for invar in x
                for outvar in output_variables
            ]
        )
        assert np.allclose(flat_out, check.flatten())

    @parameterized.expand(testcase)
    def test_jacrev_scalar(self, output_variables, f):
        print("\njac_rev (scalar)")
        out = jax.jacrev(f, argnums=1)(t_eval[k], x)
        print(out)
        flat_out, _ = tree_flatten(out)
        flat_out = np.array([f for f in flat_out]).flatten()
        check = np.array(
            [
                sim[outvar].sensitivities[invar][k]
                for invar in x
                for outvar in output_variables
            ]
        ).T
        assert np.allclose(flat_out, check.flatten())

    @parameterized.expand(testcase)
    def test_jacrev_vector(self, output_variables, f):
        print("\njac_rev (vector)")
        print("scalar")
        out = jax.jacrev(f, argnums=1)(t_eval[k], x)
        print("vector")
        out = jax.jacrev(f, argnums=1)(t_eval, x)
        print(out)
        flat_out, _ = tree_flatten(out)
        flat_out = np.concatenate(np.array([f for f in flat_out]), 1).T.flatten()
        check = np.array(
            [
                sim[outvar].sensitivities[invar]
                for invar in x
                for outvar in output_variables
            ]
        )
        print("Testing with output_variables: ", output_variables)
        assert np.allclose(flat_out, check.flatten())

    @parameterized.expand(testcase)
    def test_jacrev_vmap(self, output_variables, f):
        print("\njac_rev (vmap)")
        out = jax.vmap(
            jax.jacrev(f, argnums=1),
            in_axes=(0, None),
        )(t_eval, x)
        print(out)
        flat_out, _ = tree_flatten(out)
        flat_out = np.concatenate(np.array([f for f in flat_out]), 1).T.flatten()
        check = np.array(
            [
                sim[outvar].sensitivities[invar]
                for invar in x
                for outvar in output_variables
            ]
        )
        assert np.allclose(flat_out, check.flatten())

    # Get all vars (should mirror above outputs)

    @parameterized.expand(testcase)
    def test_jacfwd_scalar_getvars(self, output_variables, f):
        print("\njac_fwd (scalar) get_vars")
        out = jax.jacfwd(idaklu_solver.get_vars(f, output_variables), argnums=1)(
            t_eval[k], x
        )
        print(out)
        flat_out, _ = tree_flatten(out)
        flat_out = np.array([f for f in flat_out]).flatten()
        check = np.array(
            [
                sim[outvar].sensitivities[invar][k]
                for invar in x
                for outvar in output_variables
            ]
        ).T
        assert np.allclose(flat_out, check.flatten())

    @parameterized.expand(testcase)
    def test_jacfwd_scalar_getvar(self, output_variables, f):
        for outvar in output_variables:
            print(f"\njac_fwd (scalar) get_var: {outvar}")
            out = jax.jacfwd(idaklu_solver.get_var(f, outvar), argnums=1)(t_eval[k], x)
            print(out)
            flat_out, _ = tree_flatten(out)
            flat_out = np.array([f for f in flat_out]).flatten()
            check = np.array([sim[outvar].sensitivities[invar][k] for invar in x]).T
            assert np.allclose(
                flat_out, check.flatten()
            ), f"Got: {flat_out}\nExpected: {check}"

    @parameterized.expand(testcase)
    def test_jacfwd_vmap_getvars(self, output_variables, f):
        print("\njac_fwd (vmap) getvars")
        out = jax.vmap(
            jax.jacfwd(idaklu_solver.get_vars(f, output_variables), argnums=1),
            in_axes=(0, None),
        )(t_eval, x)
        print(out)
        flat_out, _ = tree_flatten(out)
        flat_out = np.concatenate(np.array([f for f in flat_out]), 1).T.flatten()
        check = np.array(
            [
                sim[outvar].sensitivities[invar]
                for invar in x
                for outvar in output_variables
            ]
        )
        assert np.allclose(flat_out, check.flatten())

    @parameterized.expand(testcase)
    def test_jacfwd_vmap_getvar(self, output_variables, f):
        for outvar in output_variables:
            print(f"\njac_fwd (vmap) getvar: {outvar}")
            out = jax.vmap(
                jax.jacfwd(idaklu_solver.get_var(f, outvar), argnums=1),
                in_axes=(0, None),
            )(t_eval, x)
            print(out)
            flat_out, _ = tree_flatten(out)
            flat_out = np.concatenate(np.array([f for f in flat_out]), 0).T.flatten()
            check = np.array([sim[outvar].sensitivities[invar] for invar in x])
            assert np.allclose(
                flat_out, check.flatten()
            ), f"Got: {flat_out}\nExpected: {check}"

    @parameterized.expand(testcase)
    def test_jacrev_scalar_getvars(self, output_variables, f):
        print("\njac_rev (scalar) getvars")
        out = jax.jacrev(idaklu_solver.get_vars(f, output_variables), argnums=1)(
            t_eval[k], x
        )
        print(out)
        flat_out, _ = tree_flatten(out)
        flat_out = np.array([f for f in flat_out]).flatten()
        check = np.array(
            [
                sim[outvar].sensitivities[invar][k]
                for invar in x
                for outvar in output_variables
            ]
        ).T
        assert np.allclose(flat_out, check.flatten())

    @parameterized.expand(testcase)
    def test_jacrev_scalar_getvar(self, output_variables, f):
        for outvar in output_variables:
            print(f"\njac_rev (scalar) getvar: {outvar}")
            out = jax.jacrev(idaklu_solver.get_var(f, outvar), argnums=1)(t_eval[k], x)
            flat_out, _ = tree_flatten(out)
            flat_out = np.array([f for f in flat_out]).flatten()
            check = np.array([sim[outvar].sensitivities[invar][k] for invar in x]).T
            assert np.allclose(flat_out, check.flatten())

    @parameterized.expand(testcase)
    def test_jacrev_vmap_getvars(self, output_variables, f):
        print("\njac_rev (vmap) getvars")
        out = jax.vmap(
            jax.jacrev(idaklu_solver.get_vars(f, output_variables), argnums=1),
            in_axes=(0, None),
        )(t_eval, x)
        print(out)
        flat_out, _ = tree_flatten(out)
        flat_out = np.concatenate(np.array([f for f in flat_out]), 1).T.flatten()
        check = np.array(
            [
                sim[outvar].sensitivities[invar]
                for invar in x
                for outvar in output_variables
            ]
        )
        assert np.allclose(flat_out, check.flatten())

    @parameterized.expand(testcase)
    def test_jacrev_vmap_getvar(self, output_variables, f):
        for outvar in output_variables:
            print(f"\njac_rev (vmap) getvar: {outvar}")
            out = jax.vmap(
                jax.jacrev(idaklu_solver.get_var(f, outvar), argnums=1),
                in_axes=(0, None),
            )(t_eval, x)
            print(out)
            flat_out, _ = tree_flatten(out)
            flat_out = np.array([f for f in flat_out]).flatten()
            check = np.array([sim[outvar].sensitivities[invar] for invar in x])
            assert np.allclose(flat_out, check.flatten())

    # Per variable checks

    @parameterized.expand(testcase)
    def test_grad_scalar_getvar(self, output_variables, f):
        for outvar in output_variables:
            print(f"\ngrad (scalar) getvar: {outvar}")
            out = jax.grad(
                idaklu_solver.get_var(f, outvar),
                argnums=1,
            )(
                t_eval[k], x
            )  # output should be a dictionary of inputs
            print(out)
            flat_out, _ = tree_flatten(out)
            flat_out = np.array([f for f in flat_out]).flatten()
            check = np.array([sim[outvar].sensitivities[invar][k] for invar in x])
            print("expected: ", check.flatten())
            print("got: ", flat_out)
            assert np.allclose(flat_out, check.flatten())

    @parameterized.expand(testcase)
    def test_grad_vmap_getvar(self, output_variables, f):
        for outvar in output_variables:
            print(f"\ngrad (vmap) getvars: {outvar}")
            out = jax.vmap(
                jax.grad(
                    idaklu_solver.get_var(f, outvar),
                    argnums=1,
                ),
                in_axes=(0, None),
            )(t_eval, x)
            print(out)
            flat_out, _ = tree_flatten(out)
            flat_out = np.array([f for f in flat_out]).flatten()
            check = np.array([sim[outvar].sensitivities[invar] for invar in x])
            assert np.allclose(flat_out, check.flatten())

    @parameterized.expand(testcase)
    def test_value_and_grad_scalar(self, output_variables, f):
        for outvar in output_variables:
            print(f"\nvalue_and_grad (scalar): {outvar}")
            primals, tangents = jax.value_and_grad(
                idaklu_solver.get_var(f, outvar),
                argnums=1,
            )(t_eval[k], x)
            print(primals)
            flat_p, _ = tree_flatten(primals)
            flat_p = np.array([f for f in flat_p]).flatten()
            check = np.array(sim[outvar].data[k])
            assert np.allclose(flat_p, check.flatten())
            print(tangents)
            flat_t, _ = tree_flatten(tangents)
            flat_t = np.array([f for f in flat_t]).flatten()
            check = np.array([sim[outvar].sensitivities[invar][k] for invar in x])
            assert np.allclose(flat_t, check.flatten())

    @parameterized.expand(testcase)
    def test_value_and_grad_vmap(self, output_variables, f):
        for outvar in output_variables:
            print(f"\nvalue_and_grad (vmap): {outvar}")
            primals, tangents = jax.vmap(
                jax.value_and_grad(
                    idaklu_solver.get_var(f, outvar),
                    argnums=1,
                ),
                in_axes=(0, None),
            )(t_eval, x)
            print(primals)
            flat_p, _ = tree_flatten(primals)
            flat_p = np.array([f for f in flat_p]).flatten()
            check = np.array(sim[outvar].data)
            assert np.allclose(flat_p, check.flatten())
            print(tangents)
            flat_t, _ = tree_flatten(tangents)
            flat_t = np.array([f for f in flat_t]).flatten()
            check = np.array([sim[outvar].sensitivities[invar] for invar in x])
            assert np.allclose(flat_t, check.flatten())

    @parameterized.expand(testcase)
    def test_jax_vars(self, output_variables, f):
        print("\njax_vars")
        out = idaklu_solver.jax_value()
        print(out)
        for outvar in output_variables:
            flat_out, _ = tree_flatten(out[outvar])
            flat_out = np.array([f for f in flat_out]).flatten()
            check = np.array(sim[outvar].data)
            assert np.allclose(
                flat_out, check.flatten()
            ), f"{outvar}: Got: {flat_out}\nExpected: {check}"

    @parameterized.expand(testcase)
    def test_jax_grad(self, output_variables, f):
        print("\njax_grad")
        out = idaklu_solver.jax_grad()
        print(out)
        for outvar in output_variables:
            flat_out, _ = tree_flatten(out[outvar])
            flat_out = np.array([f for f in flat_out]).flatten()
            check = np.array([sim[outvar].sensitivities[invar] for invar in x])
            assert np.allclose(
                flat_out, check.flatten()
            ), f"{outvar}: Got: {flat_out}\nExpected: {check}"

    @parameterized.expand(testcase)
    def test_grad_wrapper_sse(self, output_variables, f):
        print("\ngrad_wrapper_sse")

        # Use surrogate for experimental data
        data = sim["Terminal voltage [V]"](t_eval)

        # Define SSE function
        #
        # Note that although f returns a vector over time, sse() returns a scalar so
        # can be passed to grad() directly using vector time inputs.
        def sse(t, inputs):
            vf = idaklu_solver.get_var(f, "Terminal voltage [V]")
            return jnp.sum((vf(t_eval, inputs) - data) ** 2)

        # Create an imperfect prediction
        inputs_pred = inputs.copy()
        inputs_pred["Current function [A]"] = 0.150
        sim_pred = idaklu_solver.solve(
            model,
            t_eval,
            inputs=inputs_pred,
            calculate_sensitivities=True,
        )
        pred = sim_pred["Terminal voltage [V]"]

        # Check value against actual SSE
        sse_actual = np.sum((pred(t_eval) - data) ** 2)
        print(f"SSE: {sse(t_eval, inputs_pred)}")
        print(f"SSE-actual: {sse_actual}")
        flat_out, _ = tree_flatten(sse(t_eval, inputs_pred))
        flat_out = np.array([f for f in flat_out]).flatten()
        flat_check_val, _ = tree_flatten(sse_actual)
        assert np.allclose(
            flat_out, flat_check_val, 1e-3
        ), f"Got: {flat_out}\nExpected: {flat_check_val}"

        # Check grad against actual
        sse_grad_actual = {}
        for k, v in inputs_pred.items():
            sse_grad_actual[k] = 2 * np.sum(
                (pred(t_eval) - data) * pred.sensitivities[k]
            )
        sse_grad = jax.grad(sse, argnums=1)(t_eval, inputs_pred)
        print(f"SSE-grad: {sse_grad}")
        print(f"SSE-grad-actual: {sse_grad_actual}")
        flat_out, _ = tree_flatten(sse_grad)
        flat_out = np.array([f for f in flat_out]).flatten()
        flat_check_grad, _ = tree_flatten(sse_grad_actual)
        assert np.allclose(
            flat_out, flat_check_grad, 1e-3
        ), f"Got: {flat_out}\nExpected: {flat_check_grad}"

        # Check value_and_grad return
        sse_val, sse_grad = jax.value_and_grad(sse, argnums=1)(t_eval, inputs_pred)
        flat_sse_grad, _ = tree_flatten(sse_grad)
        flat_sse_grad = np.array([f for f in flat_sse_grad]).flatten()
        assert np.allclose(
            sse_val, flat_check_val, 1e3
        ), f"Got: {sse_val}\nExpected: {flat_check_val}"
        assert np.allclose(
            flat_sse_grad, flat_check_grad, 1e3
        ), f"Got: {sse_grad}\nExpected: {sse_grad}"

    # @parameterized.expand(testcase)
    # def test_f_jit_scalar(self, output_variables, f):
    #     print("\nf_jit")
    #     f_jit = jax.jit(f)
    #     print("f(t_eval, inputs): ", f(t_eval[k], inputs))
    #     print("f_jit(t_eval, inputs): ", f_jit(t_eval[k], inputs))
