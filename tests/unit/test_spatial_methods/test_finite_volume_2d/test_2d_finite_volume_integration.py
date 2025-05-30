#
# Tests for integration using Finite Volume method
#

import pybamm
import numpy as np

from tests import get_mesh_for_testing_2d


class TestFiniteVolumeIntegration:
    def test_definite_integral(self):
        # create discretisation
        mesh = get_mesh_for_testing_2d()
        spatial_methods = {
            "negative electrode": pybamm.FiniteVolume2D(),
            "separator": pybamm.FiniteVolume2D(),
            "positive electrode": pybamm.FiniteVolume2D(),
        }
        disc = pybamm.Discretisation(mesh, spatial_methods)
        # lengths
        ln = mesh["negative electrode"].edges_lr[-1]
        ls = mesh["separator"].edges_lr[-1] - ln
        lp = mesh["positive electrode"].edges_lr[-1] - (ln + ls)
        l_tb = mesh["negative electrode"].edges_tb[-1]

        # macroscale variable (lr integration)
        var = pybamm.Variable("var", domain=["negative electrode", "separator"])
        x = pybamm.SpatialVariable(
            "x", ["negative electrode", "separator"], direction="lr"
        )
        integral_eqn = pybamm.Integral(var, x)
        disc.set_variable_slices([var])
        integral_eqn_disc = disc.process_symbol(integral_eqn)
        submesh = mesh[("negative electrode", "separator")]
        constant_y = np.ones(submesh.npts_lr * submesh.npts_tb)
        assert (integral_eqn_disc.evaluate(None, constant_y) == ln + ls).all()

        var = pybamm.Variable("var", domain=["negative electrode", "separator"])
        z = pybamm.SpatialVariable(
            "z", ["negative electrode", "separator"], direction="tb"
        )
        integral_eqn_tb = pybamm.Integral(var, z)
        disc.set_variable_slices([var])
        integral_eqn_disc_tb = disc.process_symbol(integral_eqn_tb)
        submesh = mesh[("negative electrode", "separator")]
        constant_y = np.ones(submesh.npts_lr * submesh.npts_tb)
        np.testing.assert_allclose(
            integral_eqn_disc_tb.evaluate(None, constant_y), l_tb
        )

        var = pybamm.Variable(
            "var", domain=["negative electrode", "separator", "positive electrode"]
        )
        x = pybamm.SpatialVariable(
            "x",
            ["negative electrode", "separator", "positive electrode"],
            direction="lr",
        )
        integral_eqn = pybamm.Integral(var, x)
        disc.set_variable_slices([var])
        integral_eqn_disc = disc.process_symbol(integral_eqn)
        submesh = mesh[("negative electrode", "separator", "positive electrode")]
        LR, TB = np.meshgrid(submesh.edges_lr, submesh.edges_tb)
        lr = LR.flatten()
        np.testing.assert_allclose(
            integral_eqn_disc.evaluate(None, lr),
            (ln + ls + lp) ** 2 / 2,
            rtol=1e-7,
            atol=1e-6,
        )
