#
# Root of the tests module.
# Provides access to all shared functionality
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals

from .test_models.standard_model_tests import StandardModelTest, OptimisationsTest
from .shared import (
    get_mesh_for_testing,
    get_p2d_mesh_for_testing,
    get_discretisation_for_testing,
)
