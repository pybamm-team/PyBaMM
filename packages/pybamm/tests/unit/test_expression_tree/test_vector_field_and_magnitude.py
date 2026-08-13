import pytest

import pybamm


class TestVectorFieldAndMagnitude:
    def test_vector_field_and_magnitude(self, mesh_2d):
        mesh = mesh_2d
        spatial_methods = {
            "macroscale": pybamm.FiniteVolume2D(),
        }
        disc = pybamm.Discretisation(mesh, spatial_methods)
        symbol_lr = pybamm.Scalar(1)
        symbol_tb = pybamm.Scalar(2)
        vector_field = pybamm.VectorField(symbol_lr, symbol_tb)
        # using constant increases coverage due to non-simplification of constants
        one = pybamm.Constant(1, "one")
        vf_plus_one = vector_field + one
        one_plus_vf = one + vector_field
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
        assert vector_field.create_copy() == vector_field

        assert one_plus_vf_processed == pybamm.VectorField(
            pybamm.Scalar(2), pybamm.Scalar(3)
        )
        assert vf_plus_one_processed == pybamm.VectorField(
            pybamm.Scalar(2), pybamm.Scalar(3)
        )
        assert vf_processed == pybamm.VectorField(pybamm.Scalar(1), pybamm.Scalar(2))

        with pytest.raises(ValueError, match=r"applied to a vector field"):
            disc.process_symbol(pybamm.Magnitude(pybamm.Scalar(1), "lr"))

        assert negative_vf_processed == pybamm.VectorField(
            pybamm.Scalar(-1), pybamm.Scalar(-2)
        )

        thing_lr = pybamm.PrimaryBroadcast(pybamm.Scalar(1), "domain_1")
        thing_tb = pybamm.PrimaryBroadcast(pybamm.Scalar(2), "domain_2")
        with pytest.raises(ValueError, match=r"same domain"):
            pybamm.VectorField(thing_lr, thing_tb)

        vf_evaluates_on_edges = pybamm.VectorField(pybamm.Scalar(1), pybamm.Scalar(2))
        vf_evaluates_on_edges.lr_field._evaluates_on_edges = lambda _: True
        vf_evaluates_on_edges.tb_field._evaluates_on_edges = lambda _: False
        with pytest.raises(ValueError, match=r"must either"):
            vf_evaluates_on_edges.evaluates_on_edges("primary")

        assert magnitude_lr.new_copy([vector_field]) == magnitude_lr

        with pytest.raises(ValueError, match=r"Invalid direction"):
            disc.process_symbol(pybamm.Magnitude(vector_field, "asdf"))

    def test_component_and_norm_discretisation(self, mesh_2d):
        spatial_methods = {"macroscale": pybamm.FiniteVolume2D()}
        disc = pybamm.Discretisation(mesh_2d, spatial_methods)
        vector_field = pybamm.VectorField(pybamm.Scalar(3), pybamm.Scalar(4))

        comp_0 = disc.process_symbol(pybamm.Component(vector_field, 0))
        comp_1 = disc.process_symbol(pybamm.Component(vector_field, 1))
        assert comp_0.evaluate() == 3
        assert comp_1.evaluate() == 4

        norm = disc.process_symbol(pybamm.Norm(vector_field))
        assert norm.evaluate() == pytest.approx(5.0)

        with pytest.raises(
            pybamm.DiscretisationError, match=r"Component can only be applied"
        ):
            disc.process_symbol(pybamm.Component(pybamm.Scalar(1), 0))

        with pytest.raises(
            pybamm.DiscretisationError, match=r"Norm can only be applied"
        ):
            disc.process_symbol(pybamm.Norm(pybamm.Scalar(1)))

    def test_mismatched_vector_field_components(self, mesh_2d):
        spatial_methods = {"macroscale": pybamm.FiniteVolume2D()}
        disc = pybamm.Discretisation(mesh_2d, spatial_methods)
        vf2 = pybamm.VectorField(pybamm.Scalar(1), pybamm.Scalar(2))
        vf3 = pybamm.VectorField(pybamm.Scalar(1), pybamm.Scalar(2), pybamm.Scalar(3))
        with pytest.raises(
            pybamm.DiscretisationError, match=r"Cannot combine VectorFields"
        ):
            disc.process_symbol(vf2 + vf3)

    def test_json_round_trip_preserves_all_components(self):
        # A 3-component field must survive serialisation; a two-component
        # _from_json would silently drop the third and later IndexError.
        from pybamm.expression_tree.operations.serialise_kernel import decode, encode

        vf = pybamm.VectorField(pybamm.Scalar(1), pybamm.Scalar(2), pybamm.Scalar(3))
        rebuilt = decode(encode(vf))
        assert rebuilt.n_components == 3
        assert rebuilt == vf

    def test_disc_state_vector_propagation(self, mesh_2d):
        # binary and unary operators on a discretised VectorField must carry
        # the _disc_state_vector attribute over to the result
        spatial_methods = {"macroscale": pybamm.FiniteVolume2D()}
        disc = pybamm.Discretisation(mesh_2d, spatial_methods)
        vector_field = pybamm.VectorField(pybamm.Scalar(1), pybamm.Scalar(2))
        disc_vf = disc.process_symbol(vector_field)
        marker = pybamm.StateVector(slice(0, 1))
        disc_vf._disc_state_vector = marker

        one = pybamm.Constant(1, "one")
        disc_sum = disc.process_symbol(vector_field + one)
        assert disc_sum._disc_state_vector is marker

        disc_neg = disc.process_symbol(-vector_field)
        assert disc_neg._disc_state_vector is marker
