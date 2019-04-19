#
# Lead-acid LOQS model
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import pybamm

import numpy as np


class LOQS(pybamm.LeadAcidBaseModel):
    """Leading-Order Quasi-Static model for lead-acid.

    **Extends**: :class:`pybamm.LeadAcidBaseModel`

    """

    def __init__(self):
        super().__init__()

        whole_cell = ["negative electrode", "separator", "positive electrode"]
        # Variables
        c_e = pybamm.Variable("Concentration", domain=[])
        eps_n = pybamm.Variable("Negative electrode porosity", domain=[])
        eps_s = pybamm.Variable("Separator porosity", domain=[])
        eps_p = pybamm.Variable("Positive electrode porosity", domain=[])

        # Parameters
        param = pybamm.standard_parameters_lead_acid
        # Current function
        i_cell = param.current_with_time

        # ODEs
        j_n = i_cell / param.l_n
        j_p = -i_cell / param.l_p
        deps_n_dt = -param.beta_surf_n * j_n
        deps_p_dt = -param.beta_surf_p * j_p
        dc_e_dt = (
            1
            / (param.l_n * eps_n + param.l_s * eps_s + param.l_p * eps_p)
            * (
                (param.s_n - param.s_p) * i_cell
                - c_e * (param.l_n * deps_n_dt + param.l_p * deps_p_dt)
            )
        )
        self.rhs = {c_e: dc_e_dt, eps_n: deps_n_dt, eps_s: 0, eps_p: deps_p_dt}
        # Initial conditions
        self.initial_conditions = {
            c_e: param.c_e_init,
            eps_n: param.eps_n_init,
            eps_s: param.eps_s_init,
            eps_p: param.eps_p_init,
        }
        # ODE model -> no boundary conditions
        self.boundary_conditions = {}

        # Variables
        j0_n = pybamm.interface.exchange_current_density(
            c_e, domain=["negative electrode"]
        )
        j0_p = pybamm.interface.exchange_current_density(
            c_e, domain=["positive electrode"]
        )
        Phi = -param.U_n(c_e) - pybamm.Function(
            np.arcsinh, j_n / (2 * j0_n * param.l_n)
        )
        V = (
            Phi
            + param.U_p(c_e)
            - pybamm.Function(np.arcsinh, j_p / (2 * j0_p * param.l_p))
        )
        # Phis_n = pybamm.Scalar(0)
        # Phis_p = V
        # Concatenate variables
        # eps = pybamm.Concatenation(eps_n, eps_s, eps_p)
        # Phis = pybamm.Concatenation(Phis_n, pybamm.Scalar(0), Phis_p)
        # self.variables = {"c": c, "eps": eps, "Phi": Phi, "Phis": Phis, "V": V}
        self.variables = {
            "Electrolyte concentration": pybamm.Broadcast(c_e, whole_cell),
            "Porosity": pybamm.Concatenation(
                pybamm.Broadcast(eps_n, ["negative electrode"]),
                pybamm.Broadcast(eps_s, ["separator"]),
                pybamm.Broadcast(eps_p, ["positive electrode"]),
            ),
            "Negative electrode overpotential": pybamm.Broadcast(
                Phi, ["negative electrode"]
            ),
            "Positive electrode overpotential": pybamm.Broadcast(
                V, ["positive electrode"]
            ),
            "Electrolyte potential": pybamm.Broadcast(Phi, whole_cell),
            "Voltage": V,
        }

        # Terminate if concentration goes below zero
        self.events = [c_e]

        "-----------------------------------------------------------------------------"
        "Post-Processing"

        # spatial variables
        spatial_vars = pybamm.standard_spatial_vars

        # electrolyte concentration
        c_e_n = pybamm.Broadcast(1, domain=["negative electrode"])
        c_e_s = pybamm.Broadcast(1, domain=["separator"])
        c_e_p = pybamm.Broadcast(1, domain=["positive electrode"])
        c_e = pybamm.Concatenation(c_e_n, c_e_s, c_e_p)

        c_e_dim = param.c_e_typ * c_e

        # current
        current_dim = param.i_typ * param.current_with_time

        # interfacial current density
        j_n = pybamm.Broadcast(j_n, ["negative electrode"])
        j_s = pybamm.Broadcast(pybamm.Scalar(0), domain=["separator"])
        j_p = pybamm.Broadcast(j_p, ["positive electrode"])
        j = pybamm.Concatenation(j_n, j_s, j_p)

        j_dim = param.i_typ * j

        # exhange current density
        j0_n = pybamm.interface.exchange_current_density(c_e_n)
        j0_s = pybamm.Broadcast(pybamm.Scalar(0), domain=["separator"])
        j0_p = pybamm.interface.exchange_current_density(c_e_p)
        j0 = pybamm.Concatenation(j0_n, j0_s, j0_p)

        j0_dim = param.i_typ * j0

        # reaction overpotentials
        eta_r_n = pybamm.interface.inverse_butler_volmer(j_n, j0_n, param.ne_n)
        eta_r_p = pybamm.interface.inverse_butler_volmer(j_p, j0_p, param.ne_p)
        eta_r_n_av = pybamm.Integral(eta_r_n, spatial_vars.x_n) / param.l_n
        eta_r_p_av = pybamm.Integral(eta_r_p, spatial_vars.x_p) / param.l_p
        eta_r_av = eta_r_p_av - eta_r_n_av

        eta_r_n_dim = param.potential_scale * eta_r_n
        eta_r_p_dim = param.potential_scale * eta_r_p
        eta_r_n_av_dim = param.potential_scale * eta_r_n_av
        eta_r_p_av_dim = param.potential_scale * eta_r_p_av
        eta_r_av_dim = param.potential_scale * eta_r_av

        # open circuit voltage
        ocp_n = pybamm.Broadcast(param.U_n(pybamm.surf(c_e_n)), ["negative electrode"])
        ocp_p = pybamm.Broadcast(param.U_p(pybamm.surf(c_e_p)), ["positive electrode"])
        ocp_n_av = pybamm.Integral(ocp_n, spatial_vars.x_n) / param.l_n
        ocp_p_av = pybamm.Integral(ocp_p, spatial_vars.x_p) / param.l_p
        ocp_n_left = pybamm.BoundaryValue(ocp_n, "left")
        ocp_p_right = pybamm.BoundaryValue(ocp_p, "right")
        ocv_av = ocp_p_av - ocp_n_av
        ocv = ocp_p_right - ocp_n_left

        ocp_n_dim = param.U_n_ref + param.potential_scale * ocp_n
        ocp_p_dim = param.U_p_ref + param.potential_scale * ocp_p
        ocp_n_av_dim = param.U_n_ref + param.potential_scale * ocp_n_av
        ocp_p_av_dim = param.U_p_ref + param.potential_scale * ocp_p_av
        ocp_n_left_dim = param.U_n_ref + param.potential_scale * ocp_n_left
        ocp_p_right_dim = param.U_p_ref + param.potential_scale * ocp_p_right
        ocv_av_dim = ocp_p_av_dim - ocp_n_av_dim
        ocv_dim = ocp_p_right_dim - ocp_n_left_dim

        # electrolyte potential, current, ohmic losses, and concentration overpotential
        elosm = pybamm.electrolyte_current.explicit_leading_order_stefan_maxwell
        phi_e, i_e, Delta_Phi_e_av, eta_c_av = elosm(param, c_e, ocp_n, eta_r_n)
        eta_e_av = eta_c_av + Delta_Phi_e_av

        phi_e_dim = -param.U_n_ref + param.potential_scale * phi_e
        i_e_dim = param.i_typ * i_e
        eta_c_av_dim = param.potential_scale * eta_c_av
        Delta_Phi_e_av_dim = param.potential_scale * Delta_Phi_e_av
        eta_e_av_dim = param.potential_scale * eta_e_av

        # electrode potentials, current, and solid phase ohmic losses
        eloo = pybamm.electrode.explicit_leading_order_ohm
        phi_s, i_s, Delta_Phi_s_av = eloo(param, phi_e, ocp_p, eta_r_p)
        phi_s_n = phi_s.orphans[0]
        phi_s_p = phi_s.orphans[2]
        i_s_n = i_s.orphans[0]
        i_s_p = i_s.orphans[2]

        phi_s_n_dim = param.potential_scale * phi_s_n
        phi_s_p_dim = param.U_p_ref - param.U_n_ref + param.potential_scale * phi_s_p
        i_s_n_dim = param.i_typ * i_s_n
        i_s_p_dim = param.i_typ * i_s_p
        Delta_Phi_s_av_dim = param.potential_scale * Delta_Phi_s_av

        # terminal voltage
        v = ocv_av + eta_r_av

        v_dim = ocv_av_dim + eta_r_av_dim

        "-----------------------------------------------------------------------------"
        "Standard Output Variables"

        # Current
        self._variables.update(
            {
                "Total current density": param.current_with_time,
                "Negative electrode current density": i_s_n,
                "Positive electrode current density": i_s_p,
                "Electrolyte current density": i_e,
                "Interfacial current density": j,
                "Exchange current density": j0,
            }
        )

        self._variables.update(
            {
                "Total current density [A m-2]": current_dim,
                "Negative electrode current density [A m-2]": i_s_n_dim,
                "Positive electrode current density [A m-2]": i_s_p_dim,
                "Electrolyte current density [A m-2]": i_e_dim,
                "Interfacial current density [A m-2]": j_dim,
                "Exchange current density [A m-2]": j0_dim,
            }
        )

        # Voltage
        self._variables.update(
            {
                "Negative electrode open circuit potential": ocp_n,
                "Positive electrode open circuit potential": ocp_p,
                "Average negative electrode open circuit potential": ocp_n_av,
                "Average positive electrode open circuit potential": ocp_p_av,
                "Average open circuit voltage": ocv_av,
                "Measured open circuit voltage": ocv,
                "Terminal voltage": v,
            }
        )

        self._variables.update(
            {
                "Negative electrode open circuit potential [V]": ocp_n_dim,
                "Positive electrode open circuit potential [V]": ocp_p_dim,
                "Average negative electrode open circuit potential [V]": ocp_n_av_dim,
                "Average positive electrode open circuit potential [V]": ocp_p_av_dim,
                "Average open circuit voltage [V]": ocv_av_dim,
                "Measured open circuit voltage [V]": ocv_dim,
                "Terminal voltage [V]": v_dim,
            }
        )

        # Overpotential
        self._variables.update(
            {
                "Negative reaction overpotential": eta_r_n,
                "Positive reaction overpotential": eta_r_p,
                "Average negative reaction overpotential": eta_r_n_av,
                "Average positive reaction overpotential": eta_r_p_av,
                "Average reaction overpotential": eta_r_av,
                "Average electrolyte overpotential": eta_e_av,
                "Average solid phase ohmic losses": Delta_Phi_s_av,
            }
        )

        self._variables.update(
            {
                "Negative reaction overpotential [V]": eta_r_n_dim,
                "Positive reaction overpotential [V]": eta_r_p_dim,
                "Average negative reaction overpotential [V]": eta_r_n_av_dim,
                "Average positive reaction overpotential [V]": eta_r_p_av_dim,
                "Average reaction overpotential [V]": eta_r_av_dim,
                "Average electrolyte overpotential [V]": eta_e_av_dim,
                "Average solid phase ohmic losses [V]": Delta_Phi_s_av_dim,
            }
        )

        # Concentration
        self._variables.update({"Electrolyte concentration": c_e})

        self._variables.update({"Electrolyte concentration [mols m-3]": c_e_dim})

        # Potential
        self._variables.update(
            {
                "Negative electrode potential": phi_s_n,
                "Positive electrode potential": phi_s_p,
                "Electrolyte potential": phi_e,
            }
        )

        self._variables.update(
            {
                "Negative electrode potential [V]": phi_s_n_dim,
                "Positive electrode potential [V]": phi_s_p_dim,
                "Electrolyte potential [V]": phi_e_dim,
            }
        )

        "-----------------------------------------------------------------------------"
        "Additional Output Variables"

        self._variables.update(
            {
                "Average concentration overpotential": eta_c_av,
                "Average electrolyte ohmic losses": Delta_Phi_e_av,
            }
        )

        self._variables.update(
            {
                "Average concentration overpotential [V]": eta_c_av_dim,
                "Average electrolyte ohmic losses [V]": Delta_Phi_e_av_dim,
            }
        )
