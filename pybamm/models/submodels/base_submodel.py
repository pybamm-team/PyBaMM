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

    def get_fundamental_variables(self):
        """
        Returns the variables in the submodel for which a PDE must be solved to obtains
        """
        return {}

    def get_derived_variables(self, variables):
        """
        Returns variables which are derived from the fundamental variables in the model.
        """
        return {}

    def set_rhs(self, variables):
        return {}

    def set_algebraic(self, variables):
        return {}

    def set_boundary_conditions(self, variables):
        return {}

    def set_initial_conditions(self, variables):
        return {}

