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
        vf_processed = disc.process_symbol(vector_field)
        vf_plus_one_processed = disc.process_symbol(vf_plus_one)
        one_plus_vf_processed = disc.process_symbol(one_plus_vf)
        magnitude_lr_processed = disc.process_symbol(magnitude_lr)
        magnitude_tb_processed = disc.process_symbol(magnitude_tb)

        assert magnitude_lr_processed.evaluate() == 1
        assert magnitude_tb_processed.evaluate() == 2
        assert vf_plus_one_processed == pybamm.VectorField(
            pybamm.Scalar(2), pybamm.Scalar(3)
        )
        assert one_plus_vf_processed == pybamm.VectorField(
            pybamm.Scalar(2), pybamm.Scalar(3)
        )
        assert vf_processed == pybamm.VectorField(pybamm.Scalar(1), pybamm.Scalar(2))
