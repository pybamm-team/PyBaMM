#
# Root of the tests module.
# Provides access to all shared functionality
#
from .integration.test_models.standard_model_tests import (
    StandardModelTest,
    OptimisationsTest,
)
from .integration.test_models.standard_output_tests import StandardOutputTests
from .integration.test_models.standard_output_comparison import StandardOutputComparison
from .shared import (
    get_mesh_for_testing,
    get_p2d_mesh_for_testing,
    get_1p1d_mesh_for_testing,
    get_2p1d_mesh_for_testing,
    get_discretisation_for_testing,
    get_p2d_discretisation_for_testing,
    get_1p1d_discretisation_for_testing,
    get_2p1d_discretisation_for_testing,
    get_unit_2p1D_mesh_for_testing,
)
from .unit.test_models.test_submodels.standard_submodel_unit_tests import (
    StandardSubModelTests,
)
