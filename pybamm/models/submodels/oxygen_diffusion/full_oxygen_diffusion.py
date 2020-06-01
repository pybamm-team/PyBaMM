#
# Class for oxygen diffusion
#
import pybamm

from .base_oxygen_diffusion import BaseModel


def separator_and_positive_only(variable):
    """Return only the separator and positive electrode children

    Parameters
    ----------
    variable : :class:`pybamm.Concatenation`
        Concatenation of variables in negative, separator, positive

    Returns
    -------
    :class:`pybamm.Concatenation`
        Concatenation of variables in separator and positive only
    """
    _, var_s, var_p = variable.orphans
    return pybamm.Concatenation(var_s, var_p)


class Full(BaseModel):
    """Class for conservation of mass of oxygen. (Full refers to unreduced by
    asymptotic methods)
    In this model, extremely fast oxygen kinetics in the negative electrode imposes
    zero oxygen concentration there, and so the oxygen variable only lives in the
    separator and positive electrode. The boundary condition at the negative electrode/
    separator interface is homogeneous Dirichlet.

    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel
    reactions : dict
        Dictionary of reaction terms

    **Extends:** :class:`pybamm.oxygen_diffusion.BaseModel`
    """

    def __init__(self, param):
        super().__init__(param)

    def get_fundamental_variables(self):
        # Oxygen concentration (oxygen concentration is zero in the negative electrode)
        c_ox_n = pybamm.FullBroadcast(0, "negative electrode", "current collector")
        c_ox_s = pybamm.Variable(
            "Separator oxygen concentration",
            domain="separator",
            auxiliary_domains={"secondary": "current collector"},
        )
        c_ox_p = pybamm.Variable(
            "Positive oxygen concentration",
            domain="positive electrode",
            auxiliary_domains={"secondary": "current collector"},
        )
        c_ox_s_p = pybamm.Concatenation(c_ox_s, c_ox_p)
        variables = {"Separator and positive electrode oxygen concentration": c_ox_s_p}

        c_ox = pybamm.Concatenation(c_ox_n, c_ox_s, c_ox_p)
        variables.update(self._get_standard_concentration_variables(c_ox))

        return variables

    def get_coupled_variables(self, variables):

        tor = separator_and_positive_only(variables["Electrolyte tortuosity"])
        c_ox = variables["Separator and positive electrode oxygen concentration"]
        # TODO: allow charge and convection?
        v_box = pybamm.Scalar(0)

        param = self.param

        N_ox_diffusion = -tor * param.curlyD_ox * pybamm.grad(c_ox)

        N_ox = N_ox_diffusion + param.C_e * c_ox * v_box
        # Flux in the negative electrode is zero
        N_ox = pybamm.Concatenation(
            pybamm.FullBroadcast(0, "negative electrode", "current collector"), N_ox
        )

        variables.update(self._get_standard_flux_variables(N_ox))

        return variables

    def set_rhs(self, variables):

        param = self.param

        eps = separator_and_positive_only(variables["Porosity"])
        deps_dt = separator_and_positive_only(variables["Porosity change"])
        c_ox = variables["Separator and positive electrode oxygen concentration"]
        N_ox = variables["Oxygen flux"].orphans[1]

        j_ox = variables["Positive electrode oxygen interfacial current density"]
        source_terms = pybamm.Concatenation(
            pybamm.FullBroadcast(0, "separator", "current collector"),
            param.s_ox_Ox * j_ox,
        )

        self.rhs = {
            c_ox: (1 / eps)
            * (-pybamm.div(N_ox) / param.C_e + source_terms - c_ox * deps_dt)
        }

    def set_boundary_conditions(self, variables):

        c_ox = variables["Separator and positive electrode oxygen concentration"]

        self.boundary_conditions = {
            c_ox: {
                "left": (pybamm.Scalar(0), "Dirichlet"),
                "right": (pybamm.Scalar(0), "Neumann"),
            }
        }

    def set_initial_conditions(self, variables):

        c_ox = variables["Separator and positive electrode oxygen concentration"]

        self.initial_conditions = {c_ox: self.param.c_ox_init}
