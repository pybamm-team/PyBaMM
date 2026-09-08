#
# Test for adding flux boundary conditions in finite volumes class
#

import numpy as np
import pytest

import pybamm
from tests import get_mesh_for_testing


class TestFluxBoundaryConditions:
    @pytest.mark.parametrize("lbc, rbc", [(0, 0), (1.5, -3.5)])
    def test_add_flux_boundary_conditions(self, rbc, lbc):
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
                "left": (pybamm.Scalar(lbc), ("Flux", flux)),
                "right": (pybamm.Scalar(rbc), ("Flux", flux)),
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

        assert np.isclose(evaluated_flux_bc[0], lbc)
        assert np.isclose(evaluated_flux_bc[-1], rbc)
        np.testing.assert_allclose(
            gradient_symbol.evaluate(y=y_test), evaluated_flux_bc[1:-1], atol=1e-12
        )
