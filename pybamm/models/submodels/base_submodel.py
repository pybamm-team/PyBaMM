#
# Base submodel class
#

import pybamm


class BaseSubModel(pybamm.BaseFullBatteryModel):
    def __init__(self, set_of_parameters):
        super().__init__()
        self._set_of_parameters = set_of_parameters
        # Initialise empty variables (to avoid overwriting with 'None')
        self.variables = {}
