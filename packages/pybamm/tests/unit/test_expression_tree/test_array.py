#
# Tests for the Array class
#


import casadi
import numpy as np
import pytest
import sympy
from scipy.sparse import csc_matrix

import pybamm


class TestArray:
    def test_name(self):
        arr = pybamm.Array(np.array([1, 2, 3]))
        assert arr.name == "Array of shape (3, 1)"

    def test_list_entries(self):
        vect = pybamm.Array([1, 2, 3])
        np.testing.assert_array_equal(vect.entries, np.array([[1], [2], [3]]))
        vect = pybamm.Array([[1], [2], [3]])
        np.testing.assert_array_equal(vect.entries, np.array([[1], [2], [3]]))

    def test_linspace(self):
        x = np.linspace(0, 1, 100)[:, np.newaxis]
        y = pybamm.linspace(0, 1, 100)
        np.testing.assert_array_equal(x, y.entries)

    def test_meshgrid(self):
        a = np.linspace(0, 5)
        b = np.linspace(0, 3)
        A, B = np.meshgrid(a, b)
        c = pybamm.linspace(0, 5)
        d = pybamm.linspace(0, 3)
        C, D = pybamm.meshgrid(c, d)
        np.testing.assert_array_equal(A, C.entries)
        np.testing.assert_array_equal(B, D.entries)

    def test_to_equation(self):
        assert pybamm.Array([1, 2]).to_equation() == sympy.Array([[1.0], [2.0]])

    def test_to_from_json(self, mocker):
        arr = pybamm.Array(np.array([1, 2, 3]))

        json_dict = {
            "name": "Array of shape (3, 1)",
            "domains": {
                "primary": [],
                "secondary": [],
                "tertiary": [],
                "quaternary": [],
            },
            "entries": [[1.0], [2.0], [3.0]],
        }

        # array to json conversion
        created_json = arr.to_json()
        assert created_json == json_dict

        # json to array conversion
        assert pybamm.Array._from_json(created_json) == arr

    def test_to_casadi_dm_sparse(self):
        # Duplicate, unsorted entries must be summed as scipy does
        matrix = csc_matrix(
            ([1.0, 2.0, 3.0, 4.0], [2, 2, 0, 1], [0, 0, 3, 3, 4]), shape=(3, 4)
        )
        matrix.has_canonical_format = False
        converted = pybamm.expression_tree.array.to_casadi_dm(matrix)
        np.testing.assert_array_equal(converted.full(), matrix.toarray())
        assert converted.nnz() == 3
        # The input is left untouched
        assert not matrix.has_canonical_format

    @pytest.mark.parametrize(
        "entries",
        [
            np.array([1.0, 0.0, -2.5]),
            np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
            np.array(7.0),
        ],
    )
    def test_to_casadi_dm_dense(self, entries):
        converted = pybamm.expression_tree.array.to_casadi_dm(entries)
        expected = casadi.DM(entries)
        assert converted.shape == expected.shape
        assert converted.is_dense()
        np.testing.assert_array_equal(converted.full(), expected.full())
