#
# Full model of electrode employing Ohm's law
#
import pybamm

from .base_ohm import BaseModel


class Full(BaseModel):
    """This submodel implements Ohm's law for electrodes, accounting for full resistance effects.

    The model solves for the electrode potential distribution using the effective conductivity,
    which is the product of the material conductivity and the transport efficiency.

    .. math::
        \\nabla \\cdot (\\sigma_{\text{eff}} \\nabla \\phi_s) = -a_j

    where:
        - :math:`\\sigma_{\text{eff}} = \\sigma \\cdot \\tau_{\text{or}}` is the effective conductivity
        - :math:`\\phi_s` is the electrode potential
        - :math:`a_j` is the volumetric interfacial current density

    """

    def __init__(self, param, domain, options=None):
        super().__init__(param, domain, options=options)

    def build(self):
        domain, Domain = self.domain_Domain
        # potential is scaled with reference potential
        if domain == "negative":
            reference = 0
            phi_s_init = pybamm.Scalar(0)
        else:
            reference = self.param.ocv_init
            phi_s_init = self.param.ocv_init

        phi_s = pybamm.Variable(
            f"{Domain} electrode potential [V]",
            domain=f"{domain} electrode",
            auxiliary_domains={"secondary": "current collector"},
            reference=reference,
        )
        phi_s.print_name = f"phi_s_{domain[0]}"

        sigma_eff = self._get_effective_conductivity()
        sum_a_j = self._coupled_electrode_variable(
            f"Sum of {domain} electrode volumetric interfacial current densities [A.m-3]"
        )
        i_s = -sigma_eff * pybamm.grad(phi_s)

        variables = self._get_standard_potential_variables(phi_s)
        variables[f"{Domain} electrode effective conductivity"] = sigma_eff
        variables.update(self._get_standard_current_variables(i_s))
        if domain == "positive":
            variables.update(self._get_standard_whole_cell_variables(i_s, phi_s))
        self.variables.update(variables)

        # multiply by Lx**2 to improve conditioning
        self.algebraic[phi_s] = self.param.L_x**2 * (pybamm.div(i_s) + sum_a_j)
        self._set_porous_boundary_conditions(phi_s, sigma_eff)
        self.initial_conditions[phi_s] = phi_s_init
