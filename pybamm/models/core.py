#
# Core model class
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals


class BaseModel(object):
    def __init__(self):
        self.name = "Base Model"

    def __str__(self):
        return self.name

    def domains(self):
        return set([domain for variable, domain in self.variables])

    def initial_conditions(self):
        raise NotImplementedError

    def pdes_rhs(self, vars):
        raise NotImplementedError
