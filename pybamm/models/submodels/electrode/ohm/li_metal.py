#
# Subodels for a lithium metal electrode
#
import pybamm
from .base_ohm import BaseModel


class LithiumMetalBaseModel(BaseModel):
    def __init__(self, param, options=None):
        super().__init__(param, "Negative", options=options)

    def _get_li_metal_interface_variables(self, delta_phi_s, phi_s, phi_e):
        param = self.param
        pot_scale = param.potential_scale
        delta_phi_s_dim = pot_scale * delta_phi_s

        variables = {
            "Negative electrode potential drop": delta_phi_s,
            "Negative electrode potential drop [V]": delta_phi_s_dim,
            "X-averaged negative electrode ohmic losses": delta_phi_s / 2,
            "X-averaged negative electrode ohmic losses [V]": delta_phi_s_dim / 2,
            "Lithium metal interface electrode potential": phi_s,
            "Lithium metal interface electrode potential [V]": pot_scale * phi_s,
            "Lithium metal interface electrolyte potential": phi_e,
            "Lithium metal interface electrolyte potential [V]": param.U_n_ref
            + pot_scale * phi_e,
        }
        return variables


class LithiumMetalSurfaceForm(LithiumMetalBaseModel):
    """Model for potential drop across a lithium metal electrode, with a
    differential or algebraic equation for the surface potential difference

    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel
    options : dict, optional
        A dictionary of options to be passed to the model.

    **Extends:** :class:`pybamm.electrode.li_metal.LithiumMetalBaseModel`
    """

    def get_fundamental_variables(self):
        ocp_ref = self.param.U_n_ref
        pot_scale = self.param.potential_scale

        delta_phi = pybamm.Variable(
            "Lithium metal interface surface potential difference",
            domain="current collector",
        )
        variables = {
            "Lithium metal interface surface potential difference": delta_phi,
            "Lithium metal interface surface potential difference [V]": ocp_ref
            + delta_phi * pot_scale,
        }

        return variables

    def get_coupled_variables(self, variables):
        param = self.param

        i_boundary_cc = variables["Current collector current density"]
        T_n = variables["Negative current collector temperature"]
        l_n = param.l_n
        delta_phi_s = i_boundary_cc * l_n / param.sigma_n(T_n)

        phi_s_cn = variables["Negative current collector potential"]
        delta_phi = variables["Lithium metal interface surface potential difference"]

        # Potentials at the anode/separator interface
        phi_s = phi_s_cn - delta_phi_s
        phi_e = phi_s - delta_phi

        variables.update(
            self._get_li_metal_interface_variables(delta_phi_s, phi_s, phi_e)
        )
        return variables

    def set_initial_conditions(self, variables):
        delta_phi = variables["Lithium metal interface surface potential difference"]
        delta_phi_init = self.param.U_n_init

        self.initial_conditions = {delta_phi: delta_phi_init}

    def set_rhs(self, variables):
        if self.options["surface form"] == "differential":
            j_pl = variables["Lithium metal plating current density"]
            j_sei = variables["SEI interfacial current density"]
            sum_j = j_pl + j_sei

            i_cc = variables["Current collector current density"]
            delta_phi = variables[
                "Lithium metal interface surface potential difference"
            ]

            C_dl = self.param.C_dl_n

            self.rhs[delta_phi] = 1 / C_dl * (i_cc - sum_j)

    def set_algebraic(self, variables):
        if self.options["surface form"] != "differential":  # also catches "false"
            j_pl = variables["Lithium metal plating current density"]
            j_sei = variables["SEI interfacial current density"]
            sum_j = j_pl + j_sei

            i_cc = variables["Current collector current density"]
            delta_phi = variables[
                "Lithium metal interface surface potential difference"
            ]

            self.algebraic[delta_phi] = i_cc - sum_j


class LithiumMetalExplicit(LithiumMetalBaseModel):
    """Explicit model for potential drop across a lithium metal electrode.

    Parameters
    ----------
    param : parameteslackr class
        The parameters to use for this submodel
    options : dict, optional
        A dictionary of options to be passed to the model.

    **Extends:** :class:`pybamm.electrode.li_metal.LithiumMetalBaseModel`
    """

    def get_coupled_variables(self, variables):
        param = self.param

        i_boundary_cc = variables["Current collector current density"]
        T_n = variables["Negative current collector temperature"]
        l_n = param.l_n
        delta_phi_s = i_boundary_cc * l_n / param.sigma_n(T_n)

        phi_s_cn = variables["Negative current collector potential"]
        delta_phi = variables["Lithium metal interface surface potential difference"]

        phi_s = phi_s_cn - delta_phi_s
        phi_e = phi_s - delta_phi

        variables.update(
            self._get_li_metal_interface_variables(delta_phi_s, phi_s, phi_e)
        )
        return variables
