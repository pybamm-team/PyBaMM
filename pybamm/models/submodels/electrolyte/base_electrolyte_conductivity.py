#
# Base class for electrolyte conductivity
#

import pybamm


class BaseElectrolyteConductivity(pybamm.BaseSubModel):
    """Base class for conservation of charge in the electrolyte.

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

        phi_e_n_av = pybamm.average(phi_e_n)
        phi_e_s_av = pybamm.average(phi_e_s)
        phi_e_p_av = pybamm.average(phi_e_p)
        eta_e_av = phi_e_p_av - phi_e_n_av

        variables = {
            "Negative electrolyte potential": phi_e_n,
            "Negative electrolyte potential [V]": -param.U_n_ref + pot_scale * phi_e_n,
            "Separator electrolyte potential": phi_e_s,
            "Separator electrolyte potential [V]": -param.U_n_ref + pot_scale * phi_e_s,
            "Positive electrolyte potential": phi_e_p,
            "Positive electrolyte potential [V]": -param.U_n_ref + pot_scale * phi_e_p,
            "Electrolyte potential": phi_e,
            "Electrolyte potential [V]": -param.U_n_ref + pot_scale * phi_e,
            "Average negative electrolyte potential": phi_e_n_av,
            "Average negative electrolyte potential [V]": -param.U_n_ref
            + pot_scale * phi_e_n_av,
            "Average separator electrolyte potential": phi_e_s_av,
            "Average separator electrolyte potential [V]": -param.U_n_ref
            + pot_scale * phi_e_s_av,
            "Average positive electrolyte potential": phi_e_p_av,
            "Average positive electrolyte potential [V]": -param.U_n_ref
            + pot_scale * phi_e_p_av,
            "Average electrolyte overpotential": eta_e_av,
            "Average electrolyte overpotential [V]": pot_scale * eta_e_av,
        }

        return variables

    def _get_standard_current_variables(self, i_e):

        i_typ = self.param.i_typ
        variables = {
            "Electrolyte current density": i_e,
            "Electrolyte current density [A.m-2]": i_typ * i_e,
        }

        return variables

    def _get_split_overpotential(self, eta_c_av, delta_phi_e_av):

        param = self.param
        pot_scale = param.potential_scale

        variables = {
            "Average concentration overpotential": eta_c_av,
            "Average electrolyte ohmic losses": delta_phi_e_av,
            "Average concentration overpotential [V]": pot_scale * eta_c_av,
            "Average electrolyte ohmic losses [V]": pot_scale * delta_phi_e_av,
        }

        return variables

    def _get_standard_surface_potential_difference_variables(
        self, delta_phi, delta_phi_av
    ):

        pot_scale = self.param.potential_scale

        variables = {
            self._domain + " electrode surface potential difference": delta_phi,
            "Average "
            + self._domain.lower()
            + " electrode surface potential difference": delta_phi_av,
            self._domain
            + " electrode surface potential difference [V]": delta_phi * pot_scale,
            "Average "
            + self._domain.lower()
            + " electrode surface potential difference [V]": delta_phi_av * pot_scale,
        }

        return variables

    def _get_domain_potential_variables(self, phi_e, domain):

        pot_scale = self.param.potential_scale
        phi_e_av = pybamm.average(phi_e)

        variables = {
            self._domain + " electrolyte potential": phi_e,
            self._domain + " electrolyte potential [V]": phi_e * pot_scale,
            "Average " + self._domain + " electrolyte potential": phi_e_av,
            "Average "
            + self._domain
            + " electrolyte potential [V]": phi_e_av * pot_scale,
        }

        return variables

    def _get_domain_current_variables(self, i_e, domain):

        i_typ = self.param.i_typ
        i_e_av = pybamm.average(i_e)

        variables = {
            self._domain + " electrolyte current density": i_e,
            self._domain + " electrolyte current density [V]": i_e * i_typ,
            "Average " + self._domain + " electrolyte current density": i_e_av,
            "Average "
            + self._domain
            + " electrolyte current density [V]": i_e_av * i_typ,
        }

        return variables

    def _get_whole_cell_variables(self, variables):

        phi_e_n = variables["Negative electrolyte potential"]
        phi_e_s = variables["Separator electrolyte potential"]
        phi_e_p = variables["Positive electrolyte potential"]
        phi_e = pybamm.Concatenation(phi_e_n, phi_e_s, phi_e_p)
        phi_e_av = pybamm.average(phi_e)

        i_e_n = variables["Negative electrolyte current density"]
        i_e_s = variables["Separator electrolyte current density"]
        i_e_p = variables["Positive electrolyte current density"]
        i_e = pybamm.Concatenation(i_e_n, i_e_s, i_e_p)

        variables.update(self._get_standard_potential_variables(phi_e, phi_e_av))
        variables.update(self._get_standard_current_variables(i_e))

        return variables
