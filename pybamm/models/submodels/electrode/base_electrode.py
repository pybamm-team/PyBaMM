#
# Base class for electrode submodels
#
import pybamm


class BaseElectrode(pybamm.BaseSubModel):
    """Base class for electrode submodels.

    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel
    domain : str
        Either 'negative' or 'positive'
    options : dict, optional
        A dictionary of options to be passed to the model.
    set_positive_potential :  bool, optional
        If True the battery model sets the positive potential based on the current.
        If False, the potential is specified by the user. Default is True.
    """

    def __init__(self, param, domain, options=None, set_positive_potential=True):
        super().__init__(param, domain, options=options)
        self.set_positive_potential = set_positive_potential

    def _get_standard_potential_variables(self, phi_s):
        """
        A private function to obtain the standard variables which
        can be derived from the potential in the electrode.

        Parameters
        ----------
        phi_s : :class:`pybamm.Symbol`
            The potential in the electrode.

        Returns
        -------
        variables : dict
            The variables which can be derived from the potential in the
            electrode.
        """
        domain, Domain = self.domain_Domain

        phi_s_av = pybamm.x_average(phi_s)

        if self.domain == "negative":
            delta_phi_s = pybamm.boundary_value(phi_s, "left") - phi_s

        elif self.domain == "positive":
            delta_phi_s = pybamm.boundary_value(phi_s, "right") - phi_s
        delta_phi_s_av = pybamm.x_average(delta_phi_s)

        variables = {
            f"{Domain} electrode potential [V]": phi_s,
            f"X-averaged {domain} electrode potential [V]": phi_s_av,
            f"{Domain} electrode ohmic losses [V]": delta_phi_s,
            f"X-averaged {domain} electrode ohmic losses [V]": delta_phi_s_av,
        }

        if self.options.electrode_types[self.domain] == "porous":
            variables.update(
                {
                    f"Gradient of {domain} electrode potential [V.m-1]": pybamm.grad(
                        phi_s
                    ),
                }
            )

        return variables

    def _get_standard_current_variables(self, i_s):
        """
        A private function to obtain the standard variables which
        can be derived from the current in the electrode.

        Parameters
        ----------
        i_s : :class:`pybamm.Symbol`
            The current in the electrode.

        Returns
        -------
        variables : dict
            The variables which can be derived from the current in the
            electrode.
        """
        Domain = self.domain.capitalize()

        variables = {
            f"{Domain} electrode current density [A.m-2]": i_s,
        }

        return variables

    def _get_standard_current_collector_potential_variables(
        self, phi_s_cn, phi_s_cp, delta_phi_contact
    ):
        """
        A private function to obtain the standard variables which
        can be derived from the potentials in the current collector.

        Parameters
        ----------
        phi_s_cn : :class:`pybamm.Symbol`
            The potential in the negative current collector.
        phi_s_cp : :class:`pybamm.Symbol`
            The potential in the positive current collector.
        delta_phi_contact : :class:`pybamm.Symbol`
            The potential difference due to the contact resistance, if any.

        Returns
        -------
        variables : dict
            The variables which can be derived from the potential in the
            current collector.
        """
        # Local potential difference
        V_cc = phi_s_cp - phi_s_cn

        # Voltage
        # Note phi_s_cn is always zero at the negative tab
        V = pybamm.boundary_value(phi_s_cp, "positive tab")

        # Voltage is local current collector potential difference at the tabs, in 1D
        # this will be equal to the local current collector potential difference
        variables = {
            "Negative current collector potential [V]": phi_s_cn,
            "Positive current collector potential [V]": phi_s_cp,
            "Local voltage [V]": V_cc,
            "Terminal voltage [V]": V - delta_phi_contact,
            "Voltage [V]": V - delta_phi_contact,
            "Contact overpotential [V]": delta_phi_contact,
        }

        return variables

    def _get_standard_whole_cell_variables(self, variables):
        """
        A private function to obtain the whole-cell versions of the
        current variables.

        Parameters
        ----------
        variables : dict
            The variables in the whole model.

        Returns
        -------
        variables : dict
            The variables in the whole model with the whole-cell
            current variables added.
        """

        if "negative electrode" not in self.options.whole_cell_domains:
            i_s_n = None
        else:
            i_s_n = variables["Negative electrode current density [A.m-2]"]
        i_s_s = pybamm.FullBroadcast(0, ["separator"], "current collector")
        i_s_p = variables["Positive electrode current density [A.m-2]"]

        i_s = pybamm.concatenation(i_s_n, i_s_s, i_s_p)

        variables.update({"Electrode current density [A.m-2]": i_s})

        if self.set_positive_potential:
            # Get phi_s_cn from the current collector submodel and phi_s_p from the
            # electrode submodel
            phi_s_cn = variables["Negative current collector potential [V]"]
            phi_s_p = variables["Positive electrode potential [V]"]
            phi_s_cp = pybamm.boundary_value(phi_s_p, "right")
            if self.options["contact resistance"] == "true":
                param = self.param
                I = variables["Current [A]"]
                delta_phi_contact = I * param.R_contact
            else:
                delta_phi_contact = pybamm.Scalar(0)
            variables.update(
                self._get_standard_current_collector_potential_variables(
                    phi_s_cn, phi_s_cp, delta_phi_contact
                )
            )

        return variables
