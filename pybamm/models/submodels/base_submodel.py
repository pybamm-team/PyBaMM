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
        raise NotImplementedError

    def get_derived_variables(self):
        """
        Returns variables which are derived from the fundamental variables in the model.
        """
        raise NotImplementedError

    def set_equations(self):
        """
        Sets the governing equations in the model.
        """

        raise NotImplementedError
