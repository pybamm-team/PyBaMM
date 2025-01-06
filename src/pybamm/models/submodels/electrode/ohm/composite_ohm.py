#
# Composite model for Ohm's law in the electrode
#
import pybamm

from .base_ohm import BaseModel


class Composite(BaseModel):
    """An explicit composite leading and first order solution to solid phase
    current conservation with ohm's law. Note that the returned current density is
    only the leading order approximation.

    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel
    domain : str
        Either 'Negative electrode' or 'Positive electrode'
    options : dict, optional
        A dictionary of options to be passed to the model.
    """

    def __init__(self, param, domain, options=None):
        super().__init__(param, domain, options=options)

    def build(self, submodels):
        domain, Domain = self.domain_Domain
        i_boundary_cc = pybamm.CoupledVariable(
            "Current collector current density [A.m-2]",
            domain="current collector",
        )
        self.coupled_variables.update({i_boundary_cc.name: i_boundary_cc})

        # import parameters and spatial variables
        L_n = self.param.n.L
        L_p = self.param.p.L
        L_x = self.param.L_x
        x_n = pybamm.standard_spatial_vars.x_n
        x_p = pybamm.standard_spatial_vars.x_p

        tor = pybamm.CoupledVariable(
            f"X-averaged {domain} electrode transport efficiency",
            domain="current collector",
        )
        self.coupled_variables.update({tor.name: tor})
        phi_s_cn = pybamm.CoupledVariable(
            "Negative current collector potential [V]",
            domain="current collector",
        )
        self.coupled_variables.update({phi_s_cn.name: phi_s_cn})
        T = pybamm.CoupledVariable(
            f"X-averaged {domain} electrode temperature [K]",
            domain=f"{domain} electrode",
        )
        self.coupled_variables.update({T.name: T})

        sigma_eff = self.domain_param.sigma(T) * tor
        if self._domain == "negative":
            phi_s = phi_s_cn + (i_boundary_cc / sigma_eff) * (
                x_n * (x_n - 2 * L_n) / (2 * L_n)
            )
            i_s = i_boundary_cc * (1 - x_n / L_n)

        elif self.domain == "positive":
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

            const = (
                delta_phi_p_av
                + phi_e_p_av
                + (i_boundary_cc / sigma_eff) * (L_x - L_p / 3)
            )

            phi_s = const - (i_boundary_cc / sigma_eff) * (
                x_p + (x_p - L_x) ** 2 / (2 * L_p)
            )
            i_s = i_boundary_cc * (1 - (L_x - x_p) / L_p)

        variables = self._get_standard_potential_variables(phi_s)
        variables.update(self._get_standard_current_variables(i_s))

        if self.domain == "positive":
            variables.update(self._get_standard_whole_cell_variables(variables))

        phi_s = pybamm.CoupledVariable(
            f"{Domain} electrode potential [V]",
            domain=f"{domain} electrode",
        )
        self.coupled_variables.update({phi_s.name: phi_s})
        tor = pybamm.CoupledVariable(
            f"X-averaged {domain} electrode transport efficiency",
            domain="current collector",
        )
        self.coupled_variables.update({tor.name: tor})
        i_boundary_cc = pybamm.CoupledVariable(
            "Current collector current density [A.m-2]",
            domain="current collector",
        )
        self.coupled_variables.update({i_boundary_cc.name: i_boundary_cc})
        T = pybamm.CoupledVariable(
            f"X-averaged {domain} electrode temperature [K]",
            domain="current collector",
        )
        self.coupled_variables.update({T.name: T})

        if self.domain == "negative":
            lbc = (pybamm.Scalar(0), "Dirichlet")
            rbc = (pybamm.Scalar(0), "Neumann")

        elif self.domain == "positive":
            lbc = (pybamm.Scalar(0), "Neumann")
            sigma_eff = self.param.p.sigma(T) * tor
            rbc = (-i_boundary_cc / sigma_eff, "Neumann")

        self.boundary_conditions[phi_s] = {"left": lbc, "right": rbc}
        self.variables.update(variables)
