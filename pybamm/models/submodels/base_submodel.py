#
# Base submodel class
#


class BaseSubModel:
    def __init__(self, param):
        super().__init__()
        self.param = param
        # Initialise empty variables (to avoid overwriting with 'None')

        self.rhs = {}
        self.algebraic = {}
        self.boundary_conditions = {}
        self.initial_conditions = {}

        self.variables = {}

        self.events = {}

    def get_fundamental_variables(self):
        """
        Returns the variables in the submodel for which can be obtained without
        reference to any other submodels
        """
        return {}

    def get_coupled_variables(self, variables):
        """
        Returns variables which required variables in other submodels to be defined
        first
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

