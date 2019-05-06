#
# Root of the tests module.
# Provides access to all shared functionality
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals

from .unit.test_models.standard_model_tests import StandardModelTest, OptimisationsTest
from .unit.test_models.standard_output_tests import StandardOutputTests
from .shared import (
    get_mesh_for_testing,
    get_p2d_mesh_for_testing,
    get_discretisation_for_testing,
    get_p2d_discretisation_for_testing,
)
