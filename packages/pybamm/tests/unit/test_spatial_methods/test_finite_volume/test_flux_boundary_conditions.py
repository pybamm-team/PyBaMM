#
# Test for adding flux boundary conditions in finite volumes class
#

import numpy as np
import pytest

import pybamm
from tests import get_mesh_for_testing


class TestFluxBoundaryConditions:
    @pytest.mark.parametrize(
        "lbc, rbc, expected_lbc, expected_rbc",
        [
            (0, 0, 0, 0),
            (1.5, -3.5, 1.5, -3.5),
            (pybamm.Vector([1]), pybamm.Vector([-3.5]), 1, -3.5),
        ],
    )
    def test_add_flux_boundary_conditions(self, lbc, rbc, expected_lbc, expected_rbc):
        # create discretisation
        mesh = get_mesh_for_testing()
        spatial_methods = {"macroscale": pybamm.FiniteVolume()}
        disc = pybamm.Discretisation(mesh, spatial_methods)

        # Add flux boundary conditions
        whole_cell = ["negative electrode", "separator", "positive electrode"]
        var = pybamm.Variable("var", domain=whole_cell)
        flux = pybamm.Variable("flux", domain=whole_cell)
        disc.set_variable_slices([var])
        discretised_symbol = pybamm.StateVector(*disc.y_slices[var])
        bcs = {
            var: {
                "left": (pybamm.convert_to_symbol(lbc), ("Flux", flux)),
                "right": (pybamm.convert_to_symbol(rbc), ("Flux", flux)),
            }
        }

        # Test
        sp_meth = pybamm.FiniteVolume()
        sp_meth.build(mesh)
        gradient_symbol = sp_meth.gradient(var, discretised_symbol, {})
        extrapolated_symbol = sp_meth._extrapolate_gradient_to_boundaries(
            var, gradient_symbol, ["left", "right"], var.domain
        )
        flux_bc_symbol = sp_meth.add_flux_values(flux, extrapolated_symbol, bcs)

        submesh = mesh[whole_cell]
        y_test = np.linspace(0, 1, submesh.npts)
        evaluated_flux_bc = flux_bc_symbol.evaluate(y=y_test)

        assert np.isclose(evaluated_flux_bc[0], expected_lbc)
        assert np.isclose(evaluated_flux_bc[-1], expected_rbc)
        np.testing.assert_allclose(
            gradient_symbol.evaluate(y=y_test), evaluated_flux_bc[1:-1], atol=1e-12
        )

    @pytest.mark.parametrize("simplify", [True, False])
    def test_process_model_with_flux_bc(self, simplify):
        model = pybamm.BaseModel()
        u = pybamm.Variable("u", domain="domain")
        D = pybamm.Parameter("D")
        u0 = pybamm.Parameter("u0")

        params = pybamm.ParameterValues({"D": 5, "u0": 0})

        # governing equations
        N = -D * pybamm.grad(u)  # flux
        dudt = -pybamm.div(N, simplify=simplify)
        model.rhs = {u: dudt}

        # initial conditions
        model.initial_conditions = {u: u0}

        # boundary conditions
        model.boundary_conditions = {
            u: {
                "left": (pybamm.Scalar(0), ("Flux", N)),
                "right": (pybamm.Scalar(3), ("Flux", N)),
            }
        }

        params.process_model(model)
        processed_u = next(iter(model.rhs))
        divergence = (
            model.rhs[processed_u] if simplify else model.rhs[processed_u].child
        )
        assert isinstance(divergence, pybamm.Divergence)
        processed_flux = divergence.child

        for side in ("left", "right"):
            assert (
                model.boundary_conditions[processed_u][side][1][1] is not processed_flux
                if simplify
                else model.boundary_conditions[processed_u][side][1][1]
                is processed_flux
            )
