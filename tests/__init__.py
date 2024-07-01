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

from .unit.test_models.test_full_battery_models.test_lithium_ion.base_lithium_ion_tests import (
    BaseUnitTestLithiumIon,
)
from .unit.test_models.test_full_battery_models.test_lithium_ion.base_lithium_ion_half_cell_tests import (
    BaseUnitTestLithiumIonHalfCell,
)
from .integration.test_models.test_full_battery_models.test_lithium_ion.base_lithium_ion_tests import (
    BaseIntegrationTestLithiumIon,
)
from .integration.test_models.test_full_battery_models.test_lithium_ion.base_lithium_ion_half_cell_tests import (
    BaseIntegrationTestLithiumIonHalfCell,
)

from .shared import (
    get_mesh_for_testing,
    get_p2d_mesh_for_testing,
    get_size_distribution_mesh_for_testing,
    get_1p1d_mesh_for_testing,
    get_2p1d_mesh_for_testing,
    get_cylindrical_mesh_for_testing,
    get_discretisation_for_testing,
    get_p2d_discretisation_for_testing,
    get_size_distribution_disc_for_testing,
    function_test,
    multi_var_function_test,
    multi_var_function_cube_test,
    get_1p1d_discretisation_for_testing,
    get_2p1d_discretisation_for_testing,
    get_unit_2p1D_mesh_for_testing,
    get_cylindrical_discretisation_for_testing,
    get_base_model_with_battery_geometry,
    get_required_distribution_deps,
    get_optional_distribution_deps,
    get_present_optional_import_deps,
    no_internet_connection,
    assert_domain_equal,
)
