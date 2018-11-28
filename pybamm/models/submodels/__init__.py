from .electrolyte import ElectrolyteTransport


class SubModel(object):
    def initial_conditions(self):
        raise NotImplementedError

    def pdes_rhs(self, vars):
        raise NotImplementedError
