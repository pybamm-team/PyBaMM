import numpy as np
from pybamm.models import particles, electrolytes


class Model:
    """The parent model class. This contains variables and methods common to all Models
        """

    # will be useful to have current and BV stored somewhere for reference as needed
    # for RHS

    def __init__(self):
        self.sub_models = {}

    def initial_conditions(self, param, mesh):
        for i in self.sub_models.keys():
            self.sub_models[i].initial_conditions(
                list(self.sub_models.keys())[i], param, mesh
            )

    def rhs(self, operators):
        dydt = []
        for i in self.sub_models.keys():
            dydt = np.concatenate((dydt, self.sub_models[i].rhs(operators)))
        return dydt

    def update(self, y):
        # TODO: find a way to deconcatenate u independent of sub_models
        return


# Each specific model should just contain a list of the domains and the type of model
#  to use on each domain
class SPM(Model):
    # should also put an input so that user can change type of particle model
    def __init__(self):
        super().__init__()
        self.sub_models = {
            "negative particle": particles.StandardParticle(),
            "positive particle": particles.StandardParticle(),
        }


class SPMe(Model):
    def __init__(self):
        super().__init__()
        self.sub_models = {
            "negative particle": particles.StandardParticle(),
            "positive particle": particles.StandardParticle(),
            "electrolyte": electrolytes.StefanMaxwell1D(),
        }


class LeadAcid(Model):
    def __init__(self):
        super().__init__()
        self.sub_models = {"electrolyte": electrolytes.StefanMaxwell1D()}


class UserModel(Model):
    """A class so that a user can mix and match domains and submodels by only
    entering a dictionary of sub_models. All the functionality of Model is then
    retained."""

    def __init__(self, sub_models):
        super().__init__()
        self.sub_models = sub_models
