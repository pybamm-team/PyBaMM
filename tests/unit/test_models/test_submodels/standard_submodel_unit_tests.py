#
# Standard tests for the public methods of submodels
#


class StandardSubModelTests(object):

    """ Basic tests for submodels. Just tests everything runs without raising error"""

    def __init__(self, submodel, variables=None):
        # variables should be a dict of variables which are needed by the submodels
        # to run all its functions propertly
        if not variables:
            variables = {}
        self.submodel = submodel
        self.variables = variables

    def test_get_fundamental_variables(self):
        self.variables.update(self.submodel.get_fundamental_variables())

    def test_get_coupled_variables(self):
        self.variables.update(self.submodel.get_coupled_variables(self.variables))

    def test_set_rhs(self):
        self.submodel.set_rhs(self.variables)

    def test_set_algebraic(self):
        self.submodel.set_algebraic(self.variables)

    def test_set_boundary_conditions(self):
        self.submodel.set_boundary_conditions(self.variables)

    def test_set_initial_conditions(self):
        self.submodel.set_initial_conditions(self.variables)

    def test_set_events(self):
        self.submodel.set_events(self.variables)

    def test_all(self):
        self.test_get_fundamental_variables()
        self.test_get_coupled_variables()
        self.test_set_rhs()
        self.test_set_algebraic()
        self.test_set_boundary_conditions()
        self.test_set_initial_conditions()
        self.test_set_events()

