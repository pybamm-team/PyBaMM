import sys

import numpy as np
import pytest

import pybamm

if pybamm.has_jax():
    import jax
    import jax.numpy as jnp
    from jax.tree_util import tree_flatten

    k = 5  # time index for scalar tests
    t_eval = np.linspace(0, 1, 100)
    inputs = {"a": 0.1, "b": 0.2}
    output_variables = ["v", "u1", "u2"]


@pytest.fixture(autouse=True)
def clear_lru_cache():
    yield
    pybamm.IDAKLUJax._cached_solve.cache_clear()


@pytest.fixture(scope="module")
def sim():
    model = pybamm.BaseModel()
    v = pybamm.Variable("v")
    u1 = pybamm.Variable("u1")
    u2 = pybamm.Variable("u2")
    a = pybamm.InputParameter("a")
    b = pybamm.InputParameter("b")
    model.rhs = {u1: a * v, u2: b * v}
    model.algebraic = {v: 1 - v}
    model.initial_conditions = {u1: 0, u2: 0, v: 1}
    model.variables = {"v": v, "u1": u1, "u2": u2}
    disc = pybamm.Discretisation()
    disc.process_model(model)

    return pybamm.IDAKLUSolver(rtol=1e-6, atol=1e-6).solve(
        model,
        t_eval,
        inputs=inputs,
        calculate_sensitivities=True,
        t_interp=t_eval,
    )


@pytest.fixture()
def model():
    model = pybamm.BaseModel()
    v = pybamm.Variable("v")
    u1 = pybamm.Variable("u1")
    u2 = pybamm.Variable("u2")
    a = pybamm.InputParameter("a")
    b = pybamm.InputParameter("b")
    model.rhs = {u1: a * v, u2: b * v}
    model.algebraic = {v: 1 - v}
    model.initial_conditions = {u1: 0, u2: 0, v: 1}
    model.variables = {"v": v, "u1": u1, "u2": u2}
    disc = pybamm.Discretisation()
    disc.process_model(model)
    return model


@pytest.fixture()
def idaklu_solver():
    return pybamm.IDAKLUSolver(rtol=1e-6, atol=1e-6)


def no_jit(f):
    return f


if pybamm.has_jax():

    @pytest.fixture(
        params=[
            (output_variables[:1], no_jit),
            (output_variables[:1], jax.jit),
            (output_variables, no_jit),
            (output_variables, jax.jit),
        ],
        ids=["single-no-jit", "single-jit", "multiple-no-jit", "multiple-jit"],
    )
    def case(request, model):
        outvars, wrapper = request.param

        # Build jaxified solver and function
        idaklu_jax_solver = pybamm.IDAKLUSolver(rtol=1e-6, atol=1e-6).jaxify(
            model,
            t_eval,
            output_variables=outvars,
            calculate_sensitivities=True,
            t_interp=t_eval,
        )
        f = idaklu_jax_solver.get_jaxpr()
        return outvars, idaklu_jax_solver, f, wrapper


# Check the interface throws an appropriate error if either IDAKLU or JAX not available
@pytest.mark.skipif(
    pybamm.has_jax(),
    reason="JAX is available",
)
class TestIDAKLUJax_NoJax:
    def test_instantiate_fails(self):
        with pytest.raises(ModuleNotFoundError):
            pybamm.IDAKLUJax([], [], [])


@pytest.mark.skipif(
    not pybamm.has_jax(),
    reason="JAX is not available",
)
@pytest.mark.skipif(
    sys.platform.lower().startswith("win"),
    reason="IDAKLU-Jax is experimental on Windows",
)
class TestIDAKLUJax:
    # Initialisation tests

    def test_initialise_twice(self, idaklu_solver, model):
        idaklu_jax_solver = idaklu_solver.jaxify(
            model,
            t_eval,
            output_variables=output_variables,
            calculate_sensitivities=True,
        )
        with pytest.warns(UserWarning):
            idaklu_jax_solver.jaxify(
                model,
                t_eval,
                output_variables=output_variables,
                calculate_sensitivities=True,
            )

    def test_uninitialised(self, idaklu_solver, model):
        idaklu_jax_solver = idaklu_solver.jaxify(
            model,
            t_eval,
            output_variables=output_variables,
            calculate_sensitivities=True,
        )
        # simulate failure in initialisation
        idaklu_jax_solver.jaxpr = None
        with pytest.raises(pybamm.SolverError):
            idaklu_jax_solver.get_jaxpr()
        with pytest.raises(pybamm.SolverError):
            idaklu_jax_solver.jax_value()
        with pytest.raises(pybamm.SolverError):
            idaklu_jax_solver.jax_grad()

    def test_no_output_variables(self, idaklu_solver, model):
        with pytest.raises(pybamm.SolverError):
            idaklu_solver.jaxify(
                model,
                t_eval,
            )

    def test_no_inputs(self):
        # Regenerate model with no inputs
        model = pybamm.BaseModel()
        v = pybamm.Variable("v")
        u1 = pybamm.Variable("u1")
        u2 = pybamm.Variable("u2")
        model.rhs = {u1: 0.1 * v, u2: 0.2 * v}
        model.algebraic = {v: 1 - v}
        model.initial_conditions = {u1: 0, u2: 0, v: 1}
        model.variables = {"v": v, "u1": u1, "u2": u2}
        t_eval = np.linspace(0, 1, 100)
        idaklu_solver = pybamm.IDAKLUSolver(rtol=1e-6, atol=1e-6)
        # Regenerate surrogate data
        sim = idaklu_solver.solve(model, t_eval, t_interp=t_eval)
        idaklu_jax_solver = idaklu_solver.jaxify(
            model,
            t_eval,
            output_variables=output_variables,
            t_interp=t_eval,
        )
        f = idaklu_jax_solver.get_jaxpr()
        # Check that evaluation can occur (and is correct) with no inputs
        out = f(t_eval)
        np.testing.assert_allclose(
            out, np.array([sim[outvar](t_eval) for outvar in output_variables]).T
        )

    # Scalar evaluation

    def test_f_scalar(self, case, sim):
        output_variables, _, f, wrapper = case
        out = wrapper(f)(t_eval[k], inputs)
        np.testing.assert_allclose(
            out, np.array([sim[outvar](t_eval[k]) for outvar in output_variables]).T
        )

    def test_f_vector(self, case, sim):
        output_variables, _, f, wrapper = case
        out = wrapper(f)(t_eval, inputs)
        np.testing.assert_allclose(
            out, np.array([sim[outvar](t_eval) for outvar in output_variables]).T
        )

    def test_f_vmap(self, case, sim):
        output_variables, _, f, wrapper = case
        out = wrapper(jax.vmap(f, in_axes=(0, None)))(t_eval, inputs)
        np.testing.assert_allclose(
            out, np.array([sim[outvar](t_eval) for outvar in output_variables]).T
        )

    def test_f_batch_over_inputs(self, case):
        _, _, f, wrapper = case
        inputs_mock = np.array([1.0, 2.0, 3.0])
        with pytest.raises(NotImplementedError):
            wrapper(jax.vmap(f, in_axes=(None, 0)))(t_eval, inputs_mock)

    # Get all vars (should mirror test_f_* [above])

    def test_getvars_call_signature(self, case):
        output_variables, idaklu_jax_solver, f, wrapper = case
        if wrapper == jax.jit:
            return  # test does not involve a JAX expression
        with pytest.raises(ValueError):
            idaklu_jax_solver.get_vars()  # no variable name specified
        idaklu_jax_solver.get_vars(output_variables)  # (okay)
        idaklu_jax_solver.get_vars(f, output_variables)  # (okay)
        with pytest.raises(ValueError):
            idaklu_jax_solver.get_vars(1, 2, 3)  # too many arguments

    def test_getvars_scalar(self, case, sim):
        output_variables, idaklu_jax_solver, _, wrapper = case
        out = wrapper(idaklu_jax_solver.get_vars(output_variables))(t_eval[k], inputs)
        np.testing.assert_allclose(
            out, np.array([sim[outvar](t_eval[k]) for outvar in output_variables]).T
        )

    def test_getvars_vector(self, case, sim):
        output_variables, idaklu_jax_solver, _, wrapper = case
        out = wrapper(idaklu_jax_solver.get_vars(output_variables))(t_eval, inputs)
        np.testing.assert_allclose(
            out, np.array([sim[outvar](t_eval) for outvar in output_variables]).T
        )

    def test_getvars_vector_array(self, case, sim):
        output_variables, idaklu_jax_solver, _, wrapper = case
        if wrapper == jax.jit:
            return  # test does not involve a JAX expression
        array = np.array([sim[outvar](t_eval) for outvar in output_variables]).T
        out = idaklu_jax_solver.get_vars(array, output_variables)
        np.testing.assert_allclose(out, array)

    def test_getvars_vmap(self, case, sim):
        output_variables, idaklu_jax_solver, _, wrapper = case
        out = wrapper(
            jax.vmap(
                idaklu_jax_solver.get_vars(output_variables),
                in_axes=(0, None),
            ),
        )(t_eval, inputs)
        np.testing.assert_allclose(
            out, np.array([sim[outvar](t_eval) for outvar in output_variables]).T
        )

    # Isolate single output variable

    def test_getvar_call_signature(self, case):
        output_variables, idaklu_jax_solver, f, wrapper = case
        if wrapper == jax.jit:
            return  # test does not involve a JAX expression
        with pytest.raises(ValueError):
            idaklu_jax_solver.get_var()  # no variable name specified
        idaklu_jax_solver.get_var(output_variables[0])  # (okay)
        idaklu_jax_solver.get_var(f, output_variables[0])  # (okay)
        with pytest.raises(ValueError):
            idaklu_jax_solver.get_var(1, 2, 3)  # too many arguments

    def test_getvar_scalar_float_jaxpr(self, case, sim):
        output_variables, idaklu_jax_solver, _, wrapper = case
        # Per variable checks using the default JAX expression (self.jaxpr)
        for outvar in output_variables:
            out = wrapper(idaklu_jax_solver.get_var(outvar))(float(t_eval[k]), inputs)
            np.testing.assert_allclose(out, sim[outvar](float(t_eval[k])))

    def test_getvar_scalar_float_f(self, case, sim):
        output_variables, idaklu_jax_solver, f, wrapper = case
        # Per variable checks using a provided JAX expression (f)
        for outvar in output_variables:
            out = wrapper(idaklu_jax_solver.get_var(f, outvar))(
                float(t_eval[k]), inputs
            )
            np.testing.assert_allclose(out, sim[outvar](float(t_eval[k])))

    def test_getvar_scalar_jaxpr(self, case, sim):
        output_variables, idaklu_jax_solver, _, wrapper = case
        # Per variable checks using the default JAX expression (self.jaxpr)
        for outvar in output_variables:
            out = wrapper(idaklu_jax_solver.get_var(outvar))(t_eval[k], inputs)
            np.testing.assert_allclose(out, sim[outvar](t_eval[k]))

    def test_getvar_scalar_f(self, case, sim):
        output_variables, idaklu_jax_solver, _, wrapper = case
        # Per variable checks using a provided JAX expression (f)
        for outvar in output_variables:
            out = wrapper(idaklu_jax_solver.get_var(outvar))(t_eval[k], inputs)
            np.testing.assert_allclose(out, sim[outvar](t_eval[k]))

    def test_getvar_vector_jaxpr(self, case, sim):
        output_variables, idaklu_jax_solver, _, wrapper = case
        # Per variable checks using the default JAX expression (self.jaxpr)
        for outvar in output_variables:
            out = wrapper(idaklu_jax_solver.get_var(outvar))(t_eval, inputs)
            np.testing.assert_allclose(out, sim[outvar](t_eval))

    def test_getvar_vector_f(self, case, sim):
        output_variables, idaklu_jax_solver, f, wrapper = case
        # Per variable checks using a provided JAX expression (f)
        for outvar in output_variables:
            out = wrapper(idaklu_jax_solver.get_var(f, outvar))(t_eval, inputs)
            np.testing.assert_allclose(out, sim[outvar](t_eval))

    def test_getvar_vector_array(self, case, sim):
        output_variables, idaklu_jax_solver, _, wrapper = case
        # Per variable checks using a provided np.ndarray
        if wrapper == jax.jit:
            return  # test does not involve a JAX expression
        array = np.array([sim[outvar](t_eval) for outvar in output_variables]).T
        for outvar in output_variables:
            out = idaklu_jax_solver.get_var(array, outvar)
            np.testing.assert_allclose(out, sim[outvar](t_eval))

    def test_getvar_vmap(self, case, sim):
        output_variables, idaklu_jax_solver, _, wrapper = case
        for outvar in output_variables:
            out = wrapper(
                jax.vmap(
                    idaklu_jax_solver.get_var(outvar),
                    in_axes=(0, None),
                ),
            )(t_eval, inputs)
            np.testing.assert_allclose(out, sim[outvar](t_eval))

    # Differentiation rules (jacfwd)

    def test_jacfwd_scalar(self, case, sim):
        output_variables, _, f, wrapper = case
        out = wrapper(jax.jacfwd(f, argnums=1))(t_eval[k], inputs)
        flat_out, _ = tree_flatten(out)
        flat_out = np.array([f for f in flat_out]).flatten()
        check = np.array(
            [
                sim[outvar].sensitivities[invar][k]
                for invar in inputs
                for outvar in output_variables
            ]
        ).T
        np.testing.assert_allclose(flat_out, check.flatten())

    def test_jacfwd_vector(self, case, sim):
        output_variables, _, f, wrapper = case
        out = wrapper(jax.jacfwd(f, argnums=1))(t_eval, inputs)
        flat_out, _ = tree_flatten(out)
        flat_out = np.concatenate(np.array([f for f in flat_out]), 1).T.flatten()
        check = np.array(
            [
                sim[outvar].sensitivities[invar]
                for invar in inputs
                for outvar in output_variables
            ]
        )
        (
            np.testing.assert_allclose(flat_out, check.flatten()),
            f"Got: {flat_out}\nExpected: {check}",
        )

    def test_jacfwd_vmap(self, case, sim):
        output_variables, _, f, wrapper = case
        out = wrapper(
            jax.vmap(
                jax.jacfwd(f, argnums=1),
                in_axes=(0, None),
            ),
        )(t_eval, inputs)
        flat_out, _ = tree_flatten(out)
        flat_out = np.concatenate(np.array([f for f in flat_out]), 1).T.flatten()
        check = np.array(
            [
                sim[outvar].sensitivities[invar]
                for invar in inputs
                for outvar in output_variables
            ]
        )
        np.testing.assert_allclose(flat_out, check.flatten())

    def test_jacfwd_vmap_wrt_time(self, case):
        _, _, f, wrapper = case
        with pytest.raises(NotImplementedError):
            wrapper(
                jax.vmap(
                    jax.jacfwd(f, argnums=0),
                    in_axes=(0, None),
                ),
            )(t_eval, inputs)

    def test_jacfwd_batch_over_inputs(self, case):
        _, _, f, wrapper = case
        inputs_mock = np.array([1.0, 2.0, 3.0])
        with pytest.raises(NotImplementedError):
            wrapper(
                jax.vmap(
                    jax.jacfwd(f, argnums=1),
                    in_axes=(None, 0),
                ),
            )(t_eval, inputs_mock)

    # Differentiation rules (jacrev)

    def test_jacrev_scalar(self, case, sim):
        output_variables, _, f, wrapper = case
        out = wrapper(jax.jacrev(f, argnums=1))(t_eval[k], inputs)
        flat_out, _ = tree_flatten(out)
        flat_out = np.array([f for f in flat_out]).flatten()
        check = np.array(
            [
                sim[outvar].sensitivities[invar][k]
                for invar in inputs
                for outvar in output_variables
            ]
        ).T
        np.testing.assert_allclose(flat_out, check.flatten())

    def test_jacrev_vector(self, case, sim):
        output_variables, _, f, wrapper = case
        out = wrapper(jax.jacrev(f, argnums=1))(t_eval, inputs)
        flat_out, _ = tree_flatten(out)
        flat_out = np.concatenate(np.array([f for f in flat_out]), 1).T.flatten()
        check = np.array(
            [
                sim[outvar].sensitivities[invar]
                for invar in inputs
                for outvar in output_variables
            ]
        )
        np.testing.assert_allclose(flat_out, check.flatten())

    def test_jacrev_vmap(self, case, sim):
        output_variables, _, f, wrapper = case
        out = wrapper(
            jax.vmap(
                jax.jacrev(f, argnums=1),
                in_axes=(0, None),
            ),
        )(t_eval, inputs)
        flat_out, _ = tree_flatten(out)
        flat_out = np.concatenate(np.array([f for f in flat_out]), 1).T.flatten()
        check = np.array(
            [
                sim[outvar].sensitivities[invar]
                for invar in inputs
                for outvar in output_variables
            ]
        )
        np.testing.assert_allclose(flat_out, check.flatten())

    def test_jacrev_batch_over_inputs(self, case):
        _, _, f, wrapper = case
        inputs_mock = np.array([1.0, 2.0, 3.0])
        with pytest.raises(NotImplementedError):
            wrapper(
                jax.vmap(
                    jax.jacrev(f, argnums=1),
                    in_axes=(None, 0),
                ),
            )(t_eval, inputs_mock)

    # Forward differentiation rules with get_vars (multiple) and get_var (singular)

    def test_jacfwd_scalar_getvars(self, case, sim):
        output_variables, idaklu_jax_solver, _, wrapper = case
        out = wrapper(
            jax.jacfwd(
                idaklu_jax_solver.get_vars(output_variables),
                argnums=1,
            ),
        )(t_eval[k], inputs)
        flat_out, _ = tree_flatten(out)
        check = {  # Form dictionary of results from IDAKLU simulation
            invar: np.array(
                [
                    np.array(sim[outvar].sensitivities[invar][k]).squeeze()
                    for outvar in output_variables
                ]
            )
            for invar in inputs
        }
        flat_check, _ = tree_flatten(check)
        np.testing.assert_allclose(flat_out, flat_check)

    def test_jacfwd_scalar_getvar(self, case, sim):
        output_variables, idaklu_jax_solver, _, wrapper = case
        for outvar in output_variables:
            out = wrapper(
                jax.jacfwd(
                    idaklu_jax_solver.get_var(outvar),
                    argnums=1,
                ),
            )(t_eval[k], inputs)
            flat_out, _ = tree_flatten(out)
            check = {  # Form dictionary of results from IDAKLU simulation
                invar: np.array(sim[outvar].sensitivities[invar][k]).squeeze()
                for invar in inputs
            }
            flat_check, _ = tree_flatten(check)
            np.testing.assert_allclose(flat_out, flat_check)

    def test_jacfwd_vector_getvars(self, case, sim):
        output_variables, idaklu_jax_solver, _, wrapper = case
        out = wrapper(
            jax.jacfwd(
                idaklu_jax_solver.get_vars(output_variables),
                argnums=1,
            ),
        )(t_eval, inputs)
        flat_out, _ = tree_flatten(out)
        check = {  # Form dictionary of results from IDAKLU simulation
            invar: np.concatenate(
                [
                    sim[outvar].sensitivities[invar].reshape(-1, 1)
                    for outvar in output_variables
                ],
                axis=1,
            )
            for invar in inputs
        }
        flat_check, _ = tree_flatten(check)
        np.testing.assert_allclose(flat_out, flat_check)

    def test_jacfwd_vector_getvar(self, case, sim):
        output_variables, idaklu_jax_solver, _, wrapper = case
        for outvar in output_variables:
            out = wrapper(
                jax.jacfwd(
                    idaklu_jax_solver.get_var(outvar),
                    argnums=1,
                ),
            )(t_eval, inputs)
            flat_out, _ = tree_flatten(out)
            check = {  # Form dictionary of results from IDAKLU simulation
                invar: np.array(sim[outvar].sensitivities[invar]).flatten()
                for invar in inputs
            }
            flat_check, _ = tree_flatten(check)
            np.testing.assert_allclose(flat_out, flat_check)

    def test_jacfwd_vmap_getvars(self, case, sim):
        output_variables, idaklu_jax_solver, _, wrapper = case
        out = wrapper(
            jax.vmap(
                jax.jacfwd(idaklu_jax_solver.get_vars(output_variables), argnums=1),
                in_axes=(0, None),
            ),
        )(t_eval, inputs)
        flat_out, _ = tree_flatten(out)
        flat_out = np.concatenate(np.array([f for f in flat_out]), 1).T.flatten()
        check = np.array(
            [
                sim[outvar].sensitivities[invar]
                for invar in inputs
                for outvar in output_variables
            ]
        )
        np.testing.assert_allclose(flat_out, check.flatten())

    def test_jacfwd_vmap_getvar(self, case, sim):
        output_variables, idaklu_jax_solver, _, wrapper = case
        for outvar in output_variables:
            out = wrapper(
                jax.vmap(
                    jax.jacfwd(idaklu_jax_solver.get_var(outvar), argnums=1),
                    in_axes=(0, None),
                ),
            )(t_eval, inputs)
            flat_out, _ = tree_flatten(out)
            check = {  # Form dictionary of results from IDAKLU simulation
                invar: np.array(sim[outvar].sensitivities[invar]).flatten()
                for invar in inputs
            }
            flat_check, _ = tree_flatten(check)
            np.testing.assert_allclose(flat_out, flat_check)

    # Reverse differentiation rules with get_vars (multiple) and get_var (singular)

    def test_jacrev_scalar_getvars(self, case, sim):
        output_variables, idaklu_jax_solver, _, wrapper = case
        out = wrapper(
            jax.jacrev(
                idaklu_jax_solver.get_vars(output_variables),
                argnums=1,
            ),
        )(t_eval[k], inputs)
        flat_out, _ = tree_flatten(out)
        check = {  # Form dictionary of results from IDAKLU simulation
            invar: np.array(
                [
                    np.array(sim[outvar].sensitivities[invar][k]).squeeze()
                    for outvar in output_variables
                ]
            )
            for invar in inputs
        }
        flat_check, _ = tree_flatten(check)
        np.testing.assert_allclose(flat_out, flat_check)

    def test_jacrev_scalar_getvar(self, case, sim):
        output_variables, idaklu_jax_solver, _, wrapper = case
        for outvar in output_variables:
            out = wrapper(
                jax.jacrev(
                    idaklu_jax_solver.get_var(outvar),
                    argnums=1,
                ),
            )(t_eval[k], inputs)
            flat_out, _ = tree_flatten(out)
            flat_out = np.array([f for f in flat_out]).flatten()
            check = np.array(
                [sim[outvar].sensitivities[invar][k] for invar in inputs]
            ).T
            (
                np.testing.assert_allclose(flat_out, check.flatten()),
                f"Got: {flat_out}\nExpected: {check}",
            )

    def test_jacrev_vector_getvars(self, case, sim):
        output_variables, idaklu_jax_solver, _, wrapper = case
        out = wrapper(
            jax.jacrev(
                idaklu_jax_solver.get_vars(output_variables),
                argnums=1,
            ),
        )(t_eval, inputs)
        flat_out, _ = tree_flatten(out)
        check = {  # Form dictionary of results from IDAKLU simulation
            invar: np.concatenate(
                [
                    sim[outvar].sensitivities[invar].reshape(-1, 1)
                    for outvar in output_variables
                ],
                axis=1,
            )
            for invar in inputs
        }
        flat_check, _ = tree_flatten(check)
        np.testing.assert_allclose(flat_out, flat_check)

    def test_jacrev_vector_getvar(self, case, sim):
        output_variables, idaklu_jax_solver, _, wrapper = case
        for outvar in output_variables:
            out = wrapper(
                jax.jacrev(
                    idaklu_jax_solver.get_var(outvar),
                    argnums=1,
                ),
            )(t_eval, inputs)
            flat_out, _ = tree_flatten(out)
            check = {  # Form dictionary of results from IDAKLU simulation
                invar: np.array(sim[outvar].sensitivities[invar]).flatten()
                for invar in inputs
            }
            flat_check, _ = tree_flatten(check)
            np.testing.assert_allclose(flat_out, flat_check)

    def test_jacrev_vmap_getvars(self, case, sim):
        output_variables, idaklu_jax_solver, _, wrapper = case
        out = wrapper(
            jax.vmap(
                jax.jacrev(idaklu_jax_solver.get_vars(output_variables), argnums=1),
                in_axes=(0, None),
            ),
        )(t_eval, inputs)
        flat_out, _ = tree_flatten(out)
        flat_out = np.concatenate(np.array([f for f in flat_out]), 1).T.flatten()
        check = np.array(
            [
                sim[outvar].sensitivities[invar]
                for invar in inputs
                for outvar in output_variables
            ]
        )
        np.testing.assert_allclose(flat_out, check.flatten())

    def test_jacrev_vmap_getvar(self, case, sim):
        output_variables, idaklu_jax_solver, _, wrapper = case
        for outvar in output_variables:
            out = wrapper(
                jax.vmap(
                    jax.jacrev(idaklu_jax_solver.get_var(outvar), argnums=1),
                    in_axes=(0, None),
                ),
            )(t_eval, inputs)
            flat_out, _ = tree_flatten(out)
            check = {  # Form dictionary of results from IDAKLU simulation
                invar: np.array(sim[outvar].sensitivities[invar]).flatten()
                for invar in inputs
            }
            flat_check, _ = tree_flatten(check)
            np.testing.assert_allclose(flat_out, flat_check)

    # Gradient rule (takes single variable)

    def test_grad_scalar_getvar(self, case, sim):
        output_variables, idaklu_jax_solver, _, wrapper = case
        for outvar in output_variables:
            out = wrapper(
                jax.grad(
                    idaklu_jax_solver.get_var(outvar),
                    argnums=1,
                ),
            )(t_eval[k], inputs)  # output should be a dictionary of inputs
            flat_out, _ = tree_flatten(out)
            flat_out = np.array([f for f in flat_out]).flatten()
            check = np.array([sim[outvar].sensitivities[invar][k] for invar in inputs])
            np.testing.assert_allclose(flat_out, check.flatten())

    def test_grad_vmap_getvar(self, case, sim):
        output_variables, idaklu_jax_solver, _, wrapper = case
        for outvar in output_variables:
            out = wrapper(
                jax.vmap(
                    jax.grad(
                        idaklu_jax_solver.get_var(outvar),
                        argnums=1,
                    ),
                    in_axes=(0, None),
                ),
            )(t_eval, inputs)
            flat_out, _ = tree_flatten(out)
            flat_out = np.array([f for f in flat_out]).flatten()
            check = np.array([sim[outvar].sensitivities[invar] for invar in inputs])
            np.testing.assert_allclose(flat_out, check.flatten())

    # Value and gradient (takes single variable)

    def test_value_and_grad_scalar(self, case, sim):
        output_variables, idaklu_jax_solver, _, wrapper = case
        for outvar in output_variables:
            primals, tangents = wrapper(
                jax.value_and_grad(
                    idaklu_jax_solver.get_var(outvar),
                    argnums=1,
                ),
            )(t_eval[k], inputs)
            flat_p, _ = tree_flatten(primals)
            flat_p = np.array([f for f in flat_p]).flatten()
            check = np.array(sim[outvar].data[k])
            np.testing.assert_allclose(flat_p, check.flatten())
            flat_t, _ = tree_flatten(tangents)
            flat_t = np.array([f for f in flat_t]).flatten()
            check = np.array([sim[outvar].sensitivities[invar][k] for invar in inputs])
            np.testing.assert_allclose(flat_t, check.flatten())

    def test_value_and_grad_vmap(self, case, sim):
        output_variables, idaklu_jax_solver, _, wrapper = case
        for outvar in output_variables:
            primals, tangents = wrapper(
                jax.vmap(
                    jax.value_and_grad(
                        idaklu_jax_solver.get_var(outvar),
                        argnums=1,
                    ),
                    in_axes=(0, None),
                ),
            )(t_eval, inputs)
            flat_p, _ = tree_flatten(primals)
            flat_p = np.array([f for f in flat_p]).flatten()
            check = np.array(sim[outvar].data)
            np.testing.assert_allclose(flat_p, check.flatten())
            flat_t, _ = tree_flatten(tangents)
            flat_t = np.array([f for f in flat_t]).flatten()
            check = np.array([sim[outvar].sensitivities[invar] for invar in inputs])
            np.testing.assert_allclose(flat_t, check.flatten())

    # Helper functions - These return values (not jaxexprs) so cannot be JITed

    def test_jax_vars(self, case, sim):
        output_variables, idaklu_jax_solver, _, wrapper = case
        if wrapper == jax.jit:
            # Skipping test_jax_vars for jax.jit, jit not supported on helper functions
            return
        out = idaklu_jax_solver.jax_value(t_eval, inputs)
        for outvar in output_variables:
            flat_out, _ = tree_flatten(out[outvar])
            flat_out = np.array([f for f in flat_out]).flatten()
            check = np.array(sim[outvar].data)
            (
                np.testing.assert_allclose(flat_out, check.flatten()),
                f"{outvar}: Got: {flat_out}\nExpected: {check}",
            )

    def test_jax_grad(self, case, sim):
        output_variables, idaklu_jax_solver, _, wrapper = case
        if wrapper == jax.jit:
            # Skipping test_jax_grad for jax.jit, jit not supported on helper functions
            return
        out = idaklu_jax_solver.jax_grad(t_eval, inputs)
        for outvar in output_variables:
            flat_out, _ = tree_flatten(out[outvar])
            flat_out = np.array([f for f in flat_out]).flatten()
            check = np.array([sim[outvar].sensitivities[invar] for invar in inputs])
            (
                np.testing.assert_allclose(flat_out, check.flatten()),
                f"{outvar}: Got: {flat_out}\nExpected: {check}",
            )

    # Wrap jaxified expression in another function and take the gradient

    def test_grad_wrapper_sse(self, case, sim, model, idaklu_solver):
        _, idaklu_jax_solver, _, wrapper = case
        # Use surrogate for experimental data
        data = sim["v"](t_eval)

        # Define SSE function
        #
        # Note that although f returns a vector over time, sse() returns a scalar so
        # that it can be passed to grad() directly using time-vector inputs.
        def sse(t, inputs):
            vf = idaklu_jax_solver.get_var("v")
            return jnp.sum((vf(t_eval, inputs) - data) ** 2)

        # Create an imperfect prediction
        inputs_pred = inputs.copy()
        inputs_pred["a"] = 0.150
        sim_pred = idaklu_solver.solve(
            model,
            t_eval,
            inputs=inputs_pred,
            calculate_sensitivities=True,
            t_interp=t_eval,
        )
        pred = sim_pred["v"]

        # Check value against actual SSE
        sse_actual = np.sum((pred(t_eval) - data) ** 2)
        flat_out, _ = tree_flatten(sse(t_eval, inputs_pred))
        flat_out = np.array([f for f in flat_out]).flatten()
        flat_check_val, _ = tree_flatten(sse_actual)
        (
            np.testing.assert_allclose(flat_out, flat_check_val, 1e-3),
            f"Got: {flat_out}\nExpected: {flat_check_val}",
        )

        # Check grad against actual
        sse_grad_actual = {}
        for k, _ in inputs_pred.items():
            sse_grad_actual[k] = 2 * np.sum(
                (pred(t_eval) - data) * pred.sensitivities[k]
            )
        sse_grad = wrapper(jax.grad(sse, argnums=1))(t_eval, inputs_pred)
        flat_out, _ = tree_flatten(sse_grad)
        flat_out = np.array([f for f in flat_out]).flatten()
        flat_check_grad, _ = tree_flatten(sse_grad_actual)
        (
            np.testing.assert_allclose(flat_out, flat_check_grad, 1e-3),
            f"Got: {flat_out}\nExpected: {flat_check_grad}",
        )

        # Check value_and_grad return
        sse_val, sse_grad = wrapper(jax.value_and_grad(sse, argnums=1))(
            t_eval, inputs_pred
        )
        flat_sse_grad, _ = tree_flatten(sse_grad)
        flat_sse_grad = np.array([f for f in flat_sse_grad]).flatten()
        (
            np.testing.assert_allclose(sse_val, flat_check_val, 1e3),
            f"Got: {sse_val}\nExpected: {flat_check_val}",
        )
        (
            np.testing.assert_allclose(flat_sse_grad, flat_check_grad, 1e3),
            f"Got: {sse_grad}\nExpected: {sse_grad}",
        )
