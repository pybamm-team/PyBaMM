import numpy as np
import pytest

import pybamm
from tests import get_mesh_for_testing_2d


class TestFiniteVolume2D:
    def test_node_to_edge_to_node(self):
        # Create discretisation
        mesh = get_mesh_for_testing_2d()
        fin_vol = pybamm.FiniteVolume2D()
        fin_vol.build(mesh)
        n_lr = mesh["negative electrode"].npts_lr
        n_tb = mesh["negative electrode"].npts_tb

        # node to edge
        c = pybamm.StateVector(slice(0, n_lr * n_tb), domain=["negative electrode"])
        y_test = np.ones(n_lr * n_tb) * 2
        diffusivity_c_ari_lr = fin_vol.node_to_edge(
            c, method="arithmetic", direction="lr"
        )
        np.testing.assert_array_equal(
            diffusivity_c_ari_lr.evaluate(None, y_test),
            np.ones(((n_lr + 1) * (n_tb), 1)) * 2,
        )
        diffusivity_c_har_lr = fin_vol.node_to_edge(
            c, method="harmonic", direction="lr"
        )
        np.testing.assert_array_equal(
            diffusivity_c_har_lr.evaluate(None, y_test),
            np.ones(((n_lr + 1) * (n_tb), 1)) * 2,
        )
        diffusivity_c_ari_tb = fin_vol.node_to_edge(
            c, method="arithmetic", direction="tb"
        )
        np.testing.assert_array_equal(
            diffusivity_c_ari_tb.evaluate(None, y_test),
            np.ones(((n_lr) * (n_tb + 1), 1)) * 2,
        )
        diffusivity_c_har_tb = fin_vol.node_to_edge(
            c, method="harmonic", direction="tb"
        )
        np.testing.assert_array_equal(
            diffusivity_c_har_tb.evaluate(None, y_test),
            np.ones(((n_lr) * (n_tb + 1), 1)) * 2,
        )

        # bad shift key
        with pytest.raises(ValueError, match="shift key"):
            fin_vol.shift(c, "bad shift key", "arithmetic")

        with pytest.raises(ValueError, match="shift key"):
            fin_vol.shift(c, "bad shift key", "harmonic")

        # bad method
        with pytest.raises(ValueError, match="method"):
            fin_vol.shift(c, "shift key", "bad method")
