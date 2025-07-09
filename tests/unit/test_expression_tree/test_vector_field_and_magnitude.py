import pytest

import pybamm
from tests.shared import get_mesh_for_testing_2d


class TestVectorFieldAndMagnitude:
    def test_vector_field_and_magnitude(self):
        mesh = get_mesh_for_testing_2d()
        spatial_methods = {
            "macroscale": pybamm.FiniteVolume2D(),
        }
        disc = pybamm.Discretisation(mesh, spatial_methods)
        symbol_lr = pybamm.Scalar(1)
        symbol_tb = pybamm.Scalar(2)
        vector_field = pybamm.VectorField(symbol_lr, symbol_tb)
        vf_plus_one = vector_field + pybamm.Scalar(1)
        one_plus_vf = pybamm.Scalar(1) + vector_field
        magnitude_lr = pybamm.Magnitude(vector_field, "lr")
        magnitude_tb = pybamm.Magnitude(vector_field, "tb")
        negative_vf = -vector_field
        vf_processed = disc.process_symbol(vector_field)
        vf_plus_one_processed = disc.process_symbol(vf_plus_one)
        one_plus_vf_processed = disc.process_symbol(one_plus_vf)
        magnitude_lr_processed = disc.process_symbol(magnitude_lr)
        magnitude_tb_processed = disc.process_symbol(magnitude_tb)
        negative_vf_processed = disc.process_symbol(negative_vf)

        assert magnitude_lr_processed.evaluate() == 1
        assert magnitude_tb_processed.evaluate() == 2
        assert vf_plus_one_processed == pybamm.VectorField(
            pybamm.Scalar(2), pybamm.Scalar(3)
        )
        assert vf_plus_one.create_copy() == vf_plus_one
        assert one_plus_vf_processed == pybamm.VectorField(
            pybamm.Scalar(2), pybamm.Scalar(3)
        )
        assert vf_processed == pybamm.VectorField(pybamm.Scalar(1), pybamm.Scalar(2))

        with pytest.raises(ValueError, match="applied to a vector field"):
            disc.process_symbol(pybamm.Magnitude(pybamm.Scalar(1), "lr"))

        assert negative_vf_processed == pybamm.VectorField(
            pybamm.Scalar(-1), pybamm.Scalar(-2)
        )

        thing_lr = pybamm.PrimaryBroadcast(pybamm.Scalar(1), "domain_1")
        thing_tb = pybamm.PrimaryBroadcast(pybamm.Scalar(2), "domain_2")
        with pytest.raises(ValueError, match="same domain"):
            pybamm.VectorField(thing_lr, thing_tb)

        vf_evaluates_on_edges = pybamm.VectorField(pybamm.Scalar(1), pybamm.Scalar(2))
        vf_evaluates_on_edges.lr_field._evaluates_on_edges = lambda _: True
        vf_evaluates_on_edges.tb_field._evaluates_on_edges = lambda _: False
        with pytest.raises(ValueError, match="same domain"):
            vf_evaluates_on_edges.evaluates_on_edges("primary")

        assert magnitude_lr.new_copy([vector_field]) == magnitude_lr
