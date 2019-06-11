#
# Base class for electrolyte conductivity employing stefan-maxwell
#
import pybamm
import numpy as np


class BaseStefanMaxwellConductivity(pybamm.BaseSubModel):
    """Base class for conservation of charge in the electrolyte employing the
    Stefan-Maxwell constitutive equations.

    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel

    *Extends:* :class:`pybamm.BaseSubModel`
    """

    def __init__(self, param):
        super().__init__(param)

    def _get_standard_potential_variables(self, phi_e, phi_e_av):

        param = self.param
        pot_scale = param.potential_scale
        phi_e_n, phi_e_s, phi_e_p = phi_e.orphans

        eta_e_av = pybamm.average(phi_e_p) - pybamm.average(phi_e_n)

        variables = {
            "Negative electrolyte potential": phi_e_n,
            "Negative electrolyte potential [V]": -param.U_n_ref + pot_scale * phi_e_n,
            "Separator electrolyte potential": phi_e_s,
            "Separator electrolyte potential [V]": -param.U_n_ref + pot_scale * phi_e_s,
            "Positive electrolyte potential": phi_e_p,
            "Positive electrolyte potential [V]": -param.U_n_ref + pot_scale * phi_e_p,
            "Electrolyte potential": phi_e,
            "Electrolyte potential [V]": -param.U_n_ref + pot_scale * phi_e,
            "Average electrolyte overpotential": eta_e_av,
            "Average electrolyte overpotential [V]": pot_scale * eta_e_av,
        }

        return variables

    def _get_standard_current_variables(self, i_e):

        i_typ = self.param.i_typ
        variables = {
            "Electrolyte current density": i_e,
            "Electrolyte current density [A.m-2]": i_typ * i_e,
            "Average electrolyte current density": pybamm.average(i_e),
            "Average electrolyte current density [A.m-2]": pybamm.average(i_e),
        }

        return variables
