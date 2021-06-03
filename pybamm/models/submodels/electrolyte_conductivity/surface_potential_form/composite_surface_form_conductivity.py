#
# Class for composite surface form electrolyte conductivity employing stefan-maxwell
#
import pybamm

from ..composite_conductivity import Composite


class BaseModel(Composite):
    """
    Base class for composite conservation of charge in the electrolyte employing
    the Stefan-Maxwell constitutive equations employing the surface potential difference
    formulation.

    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel
    domain : str
        The domain in which the model holds
    reactions : dict
        Dictionary of reaction terms

    **Extends:** :class:`pybamm.electrolyte_conductivity.Composite`
    """

    def __init__(self, param, domain):
        super().__init__(param, domain)

    def get_fundamental_variables(self):

        if self.domain == "Negative":
            delta_phi = pybamm.standard_variables.delta_phi_n_av
        elif self.domain == "Separator":
            return {}
        elif self.domain == "Positive":
            delta_phi = pybamm.standard_variables.delta_phi_p_av

        variables = self._get_standard_surface_potential_difference_variables(delta_phi)
        return variables

    def set_initial_conditions(self, variables):

        if self.domain == "Separator":
            return

        delta_phi = variables[
            "X-averaged "
            + self.domain.lower()
            + " electrode surface potential difference"
        ]
        if self.domain == "Negative":
            delta_phi_init = self.param.U_n(self.param.c_n_init(0), self.param.T_init)
        elif self.domain == "Positive":
            delta_phi_init = self.param.U_p(self.param.c_p_init(1), self.param.T_init)

        self.initial_conditions = {delta_phi: delta_phi_init}

    def set_boundary_conditions(self, variables):
        if self.domain == "Negative":
            phi_e = variables["Electrolyte potential"]
            self.boundary_conditions = {
                phi_e: {
                    "left": (pybamm.Scalar(0), "Neumann"),
                    "right": (pybamm.Scalar(0), "Neumann"),
                }
            }


class CompositeDifferential(BaseModel):
    """
    Composite model for conservation of charge in the electrolyte employing the
    Stefan-Maxwell constitutive equations employing the surface potential difference
    formulation and where capacitance is present.

    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel


    **Extends:** :class:`BaseModel`
    """

    def __init__(self, param, domain):
        super().__init__(param, domain)

    def set_rhs(self, variables):
        if self.domain == "Separator":
            return

        param = self.param

        sum_j = variables[
            "Sum of x-averaged "
            + self.domain.lower()
            + " electrode interfacial current densities"
        ]

        sum_j_av = variables[
            "X-averaged "
            + self.domain.lower()
            + " electrode total interfacial current density"
        ]
        delta_phi = variables[
            "X-averaged "
            + self.domain.lower()
            + " electrode surface potential difference"
        ]

        if self.domain == "Negative":
            C_dl = param.C_dl_n
        elif self.domain == "Positive":
            C_dl = param.C_dl_p

        self.rhs[delta_phi] = 1 / C_dl * (sum_j_av - sum_j)


class CompositeAlgebraic(BaseModel):
    """
    Composite model for conservation of charge in the electrolyte employing the
    Stefan-Maxwell constitutive equations employing the surface potential difference
    formulation.

    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel


    **Extends:** :class:`BaseModel`
    """

    def __init__(self, param, domain):
        super().__init__(param, domain)

    def set_algebraic(self, variables):
        if self.domain == "Separator":
            return

        sum_j = variables[
            "Sum of x-averaged "
            + self.domain.lower()
            + " electrode interfacial current densities"
        ]

        sum_j_av = variables[
            "X-averaged "
            + self.domain.lower()
            + " electrode total interfacial current density"
        ]
        delta_phi = variables[
            "X-averaged "
            + self.domain.lower()
            + " electrode surface potential difference"
        ]

        self.algebraic[delta_phi] = sum_j_av - sum_j
