#
# Class for coupling Li plating and SEI models.
#
import pybamm
from ...base_submodel import BaseSubModel


class LiPlatingSEI(BaseSubModel):
    def __init__(self, param, plating_submodel: BaseSubModel, sei_submodel: BaseSubModel):
        super(LiPlatingSEI, self).__init__(None, domain=plating_submodel.domain)
        self.param = param
        self.plating = plating_submodel
        self.sei = sei_submodel

    def set_rhs(self, variables):
        domain = self.domain.lower() + " electrode"
        L_outer = variables["Outer " + domain + " sei thickness"]
        c_plated_Li = variables[f"{self.domain} electrode Li plating concentration"]

        # Coupling constant has to be scaled differently for each equation
        SEIplating1 = self.param.SEIplating1
        SEIplating2 = self.param.SEIplating2

        self.plating.rhs[c_plated_Li] = self.plating.rhs[c_plated_Li] - SEIplating1 * c_plated_Li
        self.sei.rhs[L_outer] = self.sei.rhs[L_outer] + SEIplating2 * c_plated_Li
