#
# Tests for the State Vector class
#

import numpy as np
import pytest

import pybamm


class TestStateVector:
    def test_evaluate(self):
        sv = pybamm.StateVector(slice(0, 10))
        y = np.linspace(0, 2, 19)
        np.testing.assert_array_equal(
            sv.evaluate(y=y), np.linspace(0, 1, 10)[:, np.newaxis]
        )

        # Try evaluating with a y that is too short
        y2 = np.ones(5)
        with pytest.raises(
            ValueError,
            match=r"y is too short, so value with slice is smaller than expected",
        ):
            sv.evaluate(y=y2)

    def test_evaluate_list(self):
        sv = pybamm.StateVector(slice(0, 11), slice(20, 31))
        y = np.linspace(0, 3, 31)
        np.testing.assert_allclose(
            sv.evaluate(y=y),
            np.concatenate([np.linspace(0, 1, 11), np.linspace(2, 3, 11)])[
                :, np.newaxis
            ],
            rtol=1e-7,
            atol=1e-6,
        )
        sv = pybamm.StateVector(slice(0, 11), slice(11, 20), slice(20, 31))
        y = np.linspace(0, 3, 31)
        np.testing.assert_allclose(
            sv.evaluate(y=y), y[:, np.newaxis], rtol=1e-7, atol=1e-6
        )

    def test_name(self):
        sv = pybamm.StateVector(slice(0, 10))
        assert sv.name == "y[0:10]"
        sv = pybamm.StateVector(slice(0, 10), slice(20, 30))
        assert sv.name == "y[0:10,20:30]"
        sv = pybamm.StateVector(
            slice(0, 10), slice(20, 30), slice(40, 50), slice(60, 70)
        )
        assert sv.name == "y[0:10,20:30,...,60:70]"

    def test_evaluation_array_argument_deprecated(self):
        with pytest.warns(DeprecationWarning, match=r"evaluation_array"):
            sv = pybamm.StateVector(slice(0, 3), evaluation_array=[True, False, True])
        # The slices define the entries read
        assert sv.evaluation_array == [True, True, True]

    def test_failure(self):
        with pytest.raises(TypeError, match=r"all y_slices must be slice objects"):
            pybamm.StateVector(slice(0, 10), 1)

    def test_to_from_json(self):
        sv = pybamm.StateVector(slice(0, 3))
        json_dict = {
            "name": "y[0:3]",
            "domains": {
                "primary": [],
                "secondary": [],
                "tertiary": [],
                "quaternary": [],
            },
            "y_slice": [{"start": 0, "stop": 3, "step": None}],
            "evaluation_array": [True, True, True],
        }
        assert sv.to_json() == json_dict
        assert pybamm.StateVector._from_json(json_dict) == sv

    def test_several_slices(self):
        sv = pybamm.StateVector(slice(0, 2), slice(4, 6))
        assert sv.evaluation_array == [True, True, False, False, True, True]
        assert sv.size == 4
        np.testing.assert_array_equal(
            sv.evaluate(y=np.arange(8.0)), [[0.0], [1.0], [4.0], [5.0]]
        )

    def test_identity_ignores_how_slices_are_split(self):
        assert pybamm.StateVector(slice(53)) == pybamm.StateVector(slice(0, 53))
        assert pybamm.StateVector(
            slice(0, 2), slice(2, 5), name="y"
        ) == pybamm.StateVector(slice(0, 5), name="y")
        assert pybamm.StateVector(slice(0, 2)) != pybamm.StateVector(slice(0, 3))

    @pytest.mark.parametrize(
        "y_slices",
        [
            (slice(5, 8), slice(0, 3)),
            (slice(0, 3), slice(2, 4)),
            (slice(0, 6, 2),),
            (slice(-2, 10),),
            (slice(0, -1),),
        ],
    )
    def test_invalid_slices(self, y_slices):
        with pytest.raises(pybamm.ShapeError, match=r"y_slices"):
            pybamm.StateVector(*y_slices)

    def test_empty_slice(self):
        sv = pybamm.StateVector(slice(0, 0))
        assert sv.size == 0
        assert sv.evaluate(y=np.arange(10.0)).shape == (0, 1)

    def test_unpickle_evaluation_array_state(self):
        # Pickles from before the index ranges were stored hold the full mask
        sv = pybamm.StateVector(slice(0, 2), slice(4, 6))
        state = sv.__getstate__()
        del state["_index_ranges"]
        state["_evaluation_array"] = sv.evaluation_array
        restored = pybamm.StateVector.__new__(pybamm.StateVector)
        restored.__setstate__(state)
        assert restored == sv
        np.testing.assert_array_equal(
            restored.evaluate(y=np.arange(6.0)), [[0.0], [1.0], [4.0], [5.0]]
        )

    def test_ranges_tile(self):
        ranges_tile = pybamm.expression_tree.state_vector.ranges_tile
        assert ranges_tile([(0, 2), (2, 5)], 0, 5)
        assert not ranges_tile([(0, 2), (3, 5)], 0, 5)
        assert not ranges_tile([(0, 3), (2, 5)], 0, 5)
        assert not ranges_tile([(0, 2)], 0, 5)


class TestStateVectorDot:
    def test_evaluate(self):
        sv = pybamm.StateVectorDot(slice(0, 10))
        y_dot = np.linspace(0, 2, 19)
        np.testing.assert_array_equal(
            sv.evaluate(y_dot=y_dot), np.linspace(0, 1, 10)[:, np.newaxis]
        )

        # Try evaluating with a y that is too short
        y_dot2 = np.ones(5)
        with pytest.raises(
            ValueError,
            match=r"y_dot is too short, so value with slice is smaller than expected",
        ):
            sv.evaluate(y_dot=y_dot2)

        # Try evaluating with y_dot=None
        with pytest.raises(
            TypeError,
            match=r"StateVectorDot cannot evaluate input 'y_dot=None'",
        ):
            sv.evaluate(y_dot=None)

    def test_name(self):
        sv = pybamm.StateVectorDot(slice(0, 10))
        assert sv.name == "y_dot[0:10]"

    def test_create_copy(self):
        sv = pybamm.StateVectorDot(slice(0, 1), domain="test")

        copied = sv.create_copy()

        assert type(copied) is pybamm.StateVectorDot
        assert copied == sv
        np.testing.assert_array_equal(
            copied.evaluate(y=np.array([7.0]), y_dot=np.array([3.0])), [[3.0]]
        )
