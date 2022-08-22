#
# Class for reaction driven porosity changes as a multiple of SEI/plating thicknesses
#
import pybamm
from .base_porosity import BaseModel


class PressureDriven(BaseModel):
    """Reaction-driven porosity changes as a multiple of SEI/plating thicknesses

    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel
    options : dict
        Options dictionary passed from the full model
    x_average : bool
        Whether to use x-averaged variables (SPM, SPMe, etc) or full variables (DFN)

    **Extends:** :class:`pybamm.porosity.BaseModel`
    """

    def __init__(self, param, options, x_average):
        super().__init__(param, options)
        self.x_average = x_average
    
    def get_fundamental_variables(self):
        if self.x_average is True:
            eps_n_pc = pybamm.standard_variables.eps_n_pc
            eps_s_pc = pybamm.standard_variables.eps_s_pc
            eps_p_pc = pybamm.standard_variables.eps_p_pc

            eps_n = pybamm.PrimaryBroadcast(eps_n_pc, "negative electrode")
            eps_s = pybamm.PrimaryBroadcast(eps_s_pc, "separator")
            eps_p = pybamm.PrimaryBroadcast(eps_p_pc, "positive electrode")
        else:
            eps_n = pybamm.standard_variables.eps_n
            eps_s = pybamm.standard_variables.eps_s
            eps_p = pybamm.standard_variables.eps_p
        variables = self._get_standard_porosity_variables(eps_n, eps_s, eps_p)

        return variables

    
    def get_coupled_variables(self, variables):

        if self.x_average is True:
            j_n = variables["X-averaged negative electrode interfacial current density"]
            # j_n = variables["X-averaged positive electrode interfacial current density [A.m-2]"]
            j_p = variables["X-averaged positive electrode interfacial current density"]
            eps_n = variables["X-averaged negative electrode porosity"]
            #eps_n_av = variables["X-averaged negative electrode porosity"]
           # eps_s_av = variables["X-averaged separator porosity"]
           # eps_p_av = variables["X-averaged positive electrode porosity"]
          #  eps_s = variables["X-averaged separator porosity"]
          #  eps_p = variables["X-averaged positive electrode porosity"]
            deps_s_dt = pybamm.PrimaryBroadcast(0, "current collector")
            #deps_p_dt = pybamm.PrimaryBroadcast(0, "current collector")
        else:
            j_n = variables["Negative electrode interfacial current density"]
            j_p = variables["Positive electrode interfacial current density"]
            eps_n = variables["Negative electrode porosity"]
          #  eps_s = variables["Separator porosity"]
          #  eps_p = variables["Positive electrode porosity"]
            deps_s_dt = pybamm.FullBroadcast(
                0, "separator", auxiliary_domains={"secondary": "current collector"}
            )
            #deps_p_dt = pybamm.FullBroadcast(
            #    0, "positive", auxiliary_domains={"secondary": "current collector"}
            #)
        # deps_s_dt = pybamm.PrimaryBroadcast(0, "seperator")
        # deps_s_dt = pybamm.PrimaryBroadcast(0, "current collector")
        # deps_n_dt = -self.param.beta_surf_n * j_n
        # deps_p_dt = -self.param.beta_surf_p * j_p*0

        deps_n_dt = 0.01*3*(1-eps_n)/(self.param.n.R_typ*self.param.n.c_max)*(j_n/self.param.F) 
        
        #deps_n_dt = -j_n# 3*(1-eps_n)/2.5e-06/28746.0*0.01*j_n/self.param.F #we have to also multiply by the correct value of jn
        deps_p_dt =  j_p*0
        #deps_p_dt = 0*eps_p
        #deps_s_dt = 0*eps_s
        # delta_eps_p = -3*(1-eps_p)/3.5e-06/35380.0*0.01
        variables.update(
            self._get_standard_porosity_change_variables(
                deps_n_dt, deps_s_dt, deps_p_dt
            )
        )

        return variables

    def set_rhs(self, variables):
        if self.x_average is True:
            for domain in ["negative electrode", "separator", "positive electrode"]:
                eps_av = variables["X-averaged " + domain + " porosity"]
                deps_dt_av = variables["X-averaged " + domain + " porosity change"]
                self.rhs.update({eps_av: deps_dt_av})
        else:
            eps = variables["Porosity"]
            deps_dt = variables["Porosity change"]
            self.rhs = {eps: deps_dt}

    def set_initial_conditions(self, variables):
        if self.x_average is True:
            eps_n_av = variables["X-averaged negative electrode porosity"]
            eps_s_av = variables["X-averaged separator porosity"]
            eps_p_av = variables["X-averaged positive electrode porosity"]

            self.initial_conditions = {
                eps_n_av: pybamm.x_average(self.param.n.epsilon_init),
                eps_s_av: pybamm.x_average(self.param.s.epsilon_init),
                eps_p_av: pybamm.x_average(self.param.p.epsilon_init),
            }
        else:
            eps = variables["Porosity"]
            self.initial_conditions = {eps: self.param.epsilon_init}