#
# Full model for Ohm's law in the electrode
#
import pybamm

from .base_ohm import BaseModel


class LeadingOrder(BaseModel):
    """An electrode submodel that employs Ohm's law the leading-order approximation to
    governing equations.

    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel
    domain : str
        Either 'negative' or 'positive'
    options : dict, optional
        A dictionary of options to be passed to the model.
    set_positive_potential :  bool, optional
        If True the battery model sets the positve potential based on the current.
        If False, the potential is specified by the user. Default is True.
    """

    def __init__(self, param, domain, options=None, set_positive_potential=True):
        super().__init__(
            param,
            domain,
            options=options,
            set_positive_potential=set_positive_potential,
        )

    def build(self, submodels):
        """
        Returns variables which are derived from the fundamental variables in the model.
        """
        domain, Domain = self.domain_Domain

        i_boundary_cc = pybamm.CoupledVariable(
            "Current collector current density [A.m-2]",
            domain="current collector",
        )
        self.coupled_variables.update({i_boundary_cc.name: i_boundary_cc})
        phi_s_cn = pybamm.CoupledVariable(
            "Negative current collector potential [V]",
            domain="current collector",
        )
        self.coupled_variables.update({phi_s_cn.name: phi_s_cn})
        # import parameters and spatial variables
        L_n = self.param.n.L
        L_p = self.param.p.L
        L_x = self.param.L_x
        x_n = pybamm.standard_spatial_vars.x_n
        x_p = pybamm.standard_spatial_vars.x_p

        if self.domain == "negative":
            phi_s = pybamm.PrimaryBroadcast(phi_s_cn, "negative electrode")
            i_s = i_boundary_cc * (1 - x_n / L_n)

        elif self.domain == "positive":
            # recall delta_phi = phi_s - phi_e
            delta_phi_p_av = pybamm.CoupledVariable(
                "X-averaged positive electrode surface potential difference [V]",
                domain="current collector",
            )
            self.coupled_variables.update({delta_phi_p_av.name: delta_phi_p_av})
            phi_e_p_av = pybamm.CoupledVariable(
                "X-averaged positive electrolyte potential [V]",
                domain="current collector",
            )
            self.coupled_variables.update({phi_e_p_av.name: phi_e_p_av})

            v = delta_phi_p_av + phi_e_p_av

            phi_s = pybamm.PrimaryBroadcast(v, ["positive electrode"])
            i_s = i_boundary_cc * (1 - (L_x - x_p) / L_p)

        variables = self._get_standard_potential_variables(phi_s)
        variables.update(self._get_standard_current_variables(i_s))

        if self.domain == "positive":
            variables.update(self._get_standard_whole_cell_variables(variables))

        phi_s = variables[f"{Domain} electrode potential [V]"]

        lbc = (pybamm.Scalar(0), "Neumann")
        rbc = (pybamm.Scalar(0), "Neumann")

        self.boundary_conditions[phi_s] = {"left": lbc, "right": rbc}
        self.variables.update(variables)
