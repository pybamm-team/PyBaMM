import pytest
from tests.shared import (
    SpatialMethodForTesting,
    get_mesh_for_testing,
)
import pybamm


@pytest.fixture(scope="module")
def get_discretisation_for_testing(
    xpts=None, rpts=10, mesh=None, cc_method=SpatialMethodForTesting
):
    if mesh is None:
        mesh = get_mesh_for_testing(xpts=xpts, rpts=rpts)
    spatial_methods = {
        "macroscale": SpatialMethodForTesting(),
        "negative particle": SpatialMethodForTesting(),
        "positive particle": SpatialMethodForTesting(),
        "negative particle size": SpatialMethodForTesting(),
        "positive particle size": SpatialMethodForTesting(),
        "current collector": cc_method(),
    }
    return pybamm.Discretisation(mesh, spatial_methods)
