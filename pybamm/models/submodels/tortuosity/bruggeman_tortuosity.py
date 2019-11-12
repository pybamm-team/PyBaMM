#
# Class for Bruggeman tortuosity
#
import pybamm

from .base_tortuosity import BaseModel


class Bruggeman(BaseModel):
    """Submodel for Bruggeman tortuosity

    **Extends:** :class:`pybamm.tortuosity.BaseModel`
    """

    def get_coupled_variables(self, variables):
        eps = variables[self.domain + " porosity"]

        if "Negative" in self.domain:
            brugg = self.param.b_n
        elif self.domain == "Separator":
            brugg = self.param.b_s
        if "Positive" in self.domain:
            brugg = self.param.b_p

        tor = eps ** brugg
        variables.update(self._get_standard_tortuosity_variables(tor))

        return variables
