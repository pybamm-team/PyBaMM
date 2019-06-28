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


    **Extends:** :class:`pybamm.BaseSubModel`
    """

    def __init__(self, param):
        super().__init__(param)

    def _get_standard_potential_variables(self, phi_e, phi_e_av):
        """
        A private function to obtain the standard variables which
        can be derived from the potential in the electrolyte.

        Parameters
        ----------
        phi_e : :class:`pybamm.Symbol`
            The potential in the electrolyte.
        phi_e_av : :class:`pybamm.Symbol`
            The cell-averaged potential in the electrolyte.

        Returns
        -------
        variables : dict
            The variables which can be derived from the potential in the
            electrolyte.
        """

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
        """
        A private function to obtain the standard variables which
        can be derived from the current in the electrolyte.

        Parameters
        ----------
        i_e : :class:`pybamm.Symbol`
            The current in the electrolyte.

        Returns
        -------
        variables : dict
            The variables which can be derived from the current in the
            electrolyte.
        """

        i_typ = self.param.i_typ
        variables = {
            "Electrolyte current density": i_e,
            "Electrolyte current density [A.m-2]": i_typ * i_e,
        }

        return variables

    def _get_split_overpotential(self, eta_c_av, delta_phi_e_av):
        """
        A private function to obtain the standard variables which
        can be derived from the electrode-averaged concentration
        overpotential and Ohmic losses in the electrolyte.

        Parameters
        ----------
        eta_c_av : :class:`pybamm.Symbol`
            The electrode-averaged concentration overpotential
        delta_phi_e_av: :class:`pybamm.Symbol`
            The electrode-averaged electrolyte Ohmic losses

        Returns
        -------
        variables : dict
            The variables which can be derived from the electrode-averaged
            concentration overpotential and Ohmic losses in the electrolyte
            electrolyte.
        """

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
        """
        A private function to obtain the standard variables which
        can be derived from the surface potential difference.

        Parameters
        ----------
        delta_phi_e : :class:`pybamm.Symbol`
            The surface potential difference.
        delta_phi_e_av : :class: `pybamm.Symbol`
            The electrode-averaged surface potential difference

        Returns
        -------
        variables : dict
            The variables which can be derived from the surface potential difference.
        """

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
        """
        A private function to obtain the standard variables which
        can be derived from the potential in the electrolyte split
        by domain: 'negative electrode', 'separator' and 'positive electrode'.

        Parameters
        ----------
        phi_e : :class:`pybamm.Symbol`
            The potential in the electrolyte within the domain 'domain'.
        domain : str
            The domain, either: 'Negative', 'Separator', or 'Positive'

        Returns
        -------
        variables : dict
            The variables which can be derived from the potential in the
            electrolyte in domain 'domain'.
        """

        pot_scale = self.param.potential_scale
        phi_e_av = pybamm.average(phi_e)

        variables = {
            self._domain + " electrolyte potential": phi_e,
            self._domain + " electrolyte potential [V]": phi_e * pot_scale,
            "Average " + self._domain.lower() + " electrolyte potential": phi_e_av,
            "Average "
            + self._domain.lower()
            + " electrolyte potential [V]": phi_e_av * pot_scale,
        }

        return variables

    def _get_domain_current_variables(self, i_e, domain):
        """
        A private function to obtain the standard variables which
        can be derived from the current in the electrolyte split
        by domain: 'negative electrode', 'separator' and 'positive electrode'.

        Parameters
        ----------
        i_e : :class:`pybamm.Symbol`
            The current in the electrolyte within the domain 'domain'.
        domain : str
            The domain, either: 'Negative', 'Separator', or 'Positive'

        Returns
        -------
        variables : dict
            The variables which can be derived from the current in the
            electrolyte in domain 'domain'.
        """

        i_typ = self.param.i_typ

        variables = {
            self._domain + " electrolyte current density": i_e,
            self._domain + " electrolyte current density [V]": i_e * i_typ,
        }

        return variables

    def _get_whole_cell_variables(self, variables):
        """
        A private function to obtain the potential and current concatenated
        across the whole cell. Note required 'variables' to contain the potential
        and current in the subdomains: 'negative electrode', 'separator', and
        'positive electrode'.

        Parameters
        ----------
        variables : dict
            The variables that have been set in the rest of the model.

        Returns
        -------
        variables : dict
            The variables including the whole-cell electrolyte potentials
            and currents.
        """

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
