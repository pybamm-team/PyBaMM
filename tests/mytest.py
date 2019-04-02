import pybamm
import numpy as np
import unittest
from tests import get_mesh_for_testing, get_discretisation_for_testing
from scipy.sparse import block_diag


def test_process_model_ode(self):
    # one equation
    whole_cell = ["negative electrode", "separator", "positive electrode"]
    c = pybamm.Variable("c", domain=whole_cell)
    N = pybamm.grad(c)
    model = pybamm.BaseModel()
    model.rhs = {c: pybamm.div(N)}
    model.initial_conditions = {c: pybamm.Scalar(3)}
    model.boundary_conditions = {
        N: {"left": pybamm.Scalar(0), "right": pybamm.Scalar(0)}
    }
    model.variables = {"c": c, "N": N}

    # create discretisation
    disc = get_discretisation_for_testing()
    mesh = disc.mesh

    combined_submesh = mesh.combine_submeshes(*whole_cell)
    disc.process_model(model)

    y0 = model.concatenated_initial_conditions
    np.testing.assert_array_equal(y0, 3 * np.ones_like(combined_submesh[0].nodes))
    np.testing.assert_array_equal(y0, model.concatenated_rhs.evaluate(None, y0))

    # grad and div are identity operators here
    np.testing.assert_array_equal(y0, model.variables["c"].evaluate(None, y0))
    np.testing.assert_array_equal(y0, model.variables["N"].evaluate(None, y0))

    # mass matrix is identity
    np.testing.assert_array_equal(
        np.eye(combined_submesh[0].nodes.shape[0]),
        model.mass_matrix.entries.toarray(),
    )

    # jacobian is identity
    np.testing.assert_array_equal(
        np.eye(combined_submesh[0].nodes.shape[0]),
        model.jacobian(0, y0)
    )


# one rhs equation and one algebraic
whole_cell = ["negative electrode", "separator", "positive electrode"]
c = pybamm.Variable("c", domain=whole_cell)
d = pybamm.Variable("d", domain=whole_cell)
N = pybamm.grad(c)
model = pybamm.BaseModel()
model.rhs = {c: pybamm.div(N)}
model.algebraic = {d: d - 2 * c}
model.initial_conditions = {d: pybamm.Scalar(6), c: pybamm.Scalar(3)}

model.boundary_conditions = {N: {"left": pybamm.Scalar(0), "right": pybamm.Scalar(0)}}
model.variables = {"c": c, "N": N, "d": d}

# create discretisation
disc = get_discretisation_for_testing()
mesh = disc.mesh

disc.process_model(model)
combined_submesh = mesh.combine_submeshes(*whole_cell)

y0 = model.concatenated_initial_conditions
np.testing.assert_array_equal(
    y0,
    np.concatenate(
        [
            3 * np.ones_like(combined_submesh[0].nodes),
            6 * np.ones_like(combined_submesh[0].nodes),
        ]
    ),
)

# grad and div are identity operators here
np.testing.assert_array_equal(
    y0[: combined_submesh[0].npts], model.concatenated_rhs.evaluate(None, y0)
)

np.testing.assert_array_equal(
    model.concatenated_algebraic.evaluate(None, y0),
    np.zeros_like(combined_submesh[0].nodes),
)

# mass matrix is identity upper left, zeros elsewhere
mass = block_diag(
    (
        np.eye(np.size(combined_submesh[0].nodes)),
        np.zeros(
            (np.size(combined_submesh[0].nodes), np.size(combined_submesh[0].nodes))
        ),
    )
)
np.testing.assert_array_equal(mass.toarray(), model.mass_matrix.entries.toarray())

# jacobian
jacobian = np.block(
    [
        [
            np.eye(np.size(combined_submesh[0].nodes)),
            np.zeros(
                (np.size(combined_submesh[0].nodes), np.size(combined_submesh[0].nodes))
            ),
        ],
        [
            -2 * np.eye(np.size(combined_submesh[0].nodes)),
            np.eye(np.size(combined_submesh[0].nodes)),
        ],
    ]
)
np.testing.assert_array_equal(jacobian, model.jacobian(0, y0).toarray())

if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    unittest.main()
