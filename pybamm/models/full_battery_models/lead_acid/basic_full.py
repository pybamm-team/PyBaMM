#
# Basic lead-acid model
#
import pybamm
from .base_lead_acid_model import BaseModel


class BasicFull(BaseModel):
    """Porous electrode model for lead-acid, from [2]_.

    This class differs from the :class:`pybamm.lead_acid.Full` model class in that it
    shows the whole model in a single class. This comes at the cost of flexibility in
    comparing different physical effects, and in general the main DFN class should be
    used instead.

    Parameters
    ----------
    name : str, optional
        The name of the model.

    References
    ----------
    .. [2] V Sulzer, SJ Chapman, CP Please, DA Howey, and CW Monroe. Faster lead-acid
           battery simulations from porous-electrode theory: Part II. Asymptotic
           analysis. Journal of The Electrochemical Society 166.12 (2019), A2372–A2382..


    **Extends:** :class:`pybamm.lead_acid.BaseModel`
    """

    def __init__(self, name="Basic full model"):
        super().__init__({}, name)
        # `param` is a class containing all the relevant parameters and functions for
        # this model. These are purely symbolic at this stage, and will be set by the
        # `ParameterValues` class when the model is processed.
        param = self.param

        ######################
        # Variables
        ######################
        # Variables that depend on time only are created without a domain
        Q = pybamm.Variable("Discharge capacity [A.h]")
        # Variables that vary spatially are created with a domain
        c_e_n = pybamm.Variable(
            "Negative electrolyte concentration", domain="negative electrode",
        )
        c_e_s = pybamm.Variable(
            "Separator electrolyte concentration", domain="separator",
        )
        c_e_p = pybamm.Variable(
            "Positive electrolyte concentration", domain="positive electrode",
        )
        # Concatenations combine several variables into a single variable, to simplify
        # implementing equations that hold over several domains
        c_e = pybamm.Concatenation(c_e_n, c_e_s, c_e_p)

        # Electrolyte potential
        phi_e_n = pybamm.Variable(
            "Negative electrolyte potential", domain="negative electrode",
        )
        phi_e_s = pybamm.Variable(
            "Separator electrolyte potential", domain="separator",
        )
        phi_e_p = pybamm.Variable(
            "Positive electrolyte potential", domain="positive electrode",
        )
        phi_e = pybamm.Concatenation(phi_e_n, phi_e_s, phi_e_p)

        # Electrode potential
        phi_s_n = pybamm.Variable(
            "Negative electrode potential", domain="negative electrode",
        )
        phi_s_p = pybamm.Variable(
            "Positive electrode potential", domain="positive electrode",
        )

        # Porosity
        eps_n = pybamm.Variable(
            "Negative electrode porosity", domain="negative electrode",
        )
        eps_s = pybamm.Variable("Separator porosity", domain="separator")
        eps_p = pybamm.Variable(
            "Positive electrode porosity", domain="positive electrode",
        )
        eps = pybamm.Concatenation(eps_n, eps_s, eps_p)

        # Pressure (for convection)
        pressure_n = pybamm.Variable(
            "Negative electrolyte pressure", domain="negative electrode",
        )
        pressure_p = pybamm.Variable(
            "Positive electrolyte pressure", domain="positive electrode",
        )

        # Constant temperature
        T = param.T_init

        ######################
        # Other set-up
        ######################

        # Current density
        i_cell = param.current_with_time

        # Tortuosity
        tor = pybamm.Concatenation(
            eps_n ** param.b_e_n, eps_s ** param.b_e_s, eps_p ** param.b_e_p
        )

        # Interfacial reactions
        j0_n = param.j0_n(c_e_n, T)
        j_n = (
            2
            * j0_n
            * pybamm.sinh(param.ne_n / 2 * (phi_s_n - phi_e_n - param.U_n(c_e_n, T)))
        )
        j0_p = param.j0_p(c_e_p, T)
        j_s = pybamm.PrimaryBroadcast(0, "separator")
        j_p = (
            2
            * j0_p
            * pybamm.sinh(param.ne_p / 2 * (phi_s_p - phi_e_p - param.U_p(c_e_p, T)))
        )
        j = pybamm.Concatenation(j_n, j_s, j_p)

        ######################
        # State of Charge
        ######################
        I = param.dimensional_current_with_time
        # The `rhs` dictionary contains differential equations, with the key being the
        # variable in the d/dt
        self.rhs[Q] = I * param.timescale / 3600
        # Initial conditions must be provided for the ODEs
        self.initial_conditions[Q] = pybamm.Scalar(0)

        ######################
        # Convection
        ######################
        v_n = -pybamm.grad(pressure_n)
        v_p = -pybamm.grad(pressure_p)
        l_s = pybamm.geometric_parameters.l_s
        l_n = pybamm.geometric_parameters.l_n
        x_s = pybamm.SpatialVariable("x_s", domain="separator")

        # Difference in negative and positive electrode velocities determines the
        # velocity in the separator
        v_n_right = param.beta_n * i_cell
        v_p_left = param.beta_p * i_cell
        d_v_s__dx = (v_p_left - v_n_right) / l_s

        # Simple formula for velocity in the separator
        div_V_s = -d_v_s__dx
        v_s = d_v_s__dx * (x_s - l_n) + v_n_right

        # v is the velocity in the x-direction
        # div_V is the divergence of the velocity in the yz-directions
        v = pybamm.Concatenation(v_n, v_s, v_p)
        div_V = pybamm.Concatenation(
            pybamm.PrimaryBroadcast(0, "negative electrode"),
            pybamm.PrimaryBroadcast(div_V_s, "separator"),
            pybamm.PrimaryBroadcast(0, "positive electrode"),
        )
        # Simple formula for velocity in the separator
        self.algebraic[pressure_n] = pybamm.div(v_n) - param.beta_n * j_n
        self.algebraic[pressure_p] = pybamm.div(v_p) - param.beta_p * j_p
        self.boundary_conditions[pressure_n] = {
            "left": (pybamm.Scalar(0), "Neumann"),
            "right": (pybamm.Scalar(0), "Dirichlet"),
        }
        self.boundary_conditions[pressure_p] = {
            "left": (pybamm.Scalar(0), "Dirichlet"),
            "right": (pybamm.Scalar(0), "Neumann"),
        }
        self.initial_conditions[pressure_n] = pybamm.Scalar(0)
        self.initial_conditions[pressure_p] = pybamm.Scalar(0)

        ######################
        # Current in the electrolyte
        ######################
        i_e = (param.kappa_e(c_e, T) * tor * param.gamma_e / param.C_e) * (
            param.chi(c_e) * pybamm.grad(c_e) / c_e - pybamm.grad(phi_e)
        )
        self.algebraic[phi_e] = pybamm.div(i_e) - j
        self.boundary_conditions[phi_e] = {
            "left": (pybamm.Scalar(0), "Neumann"),
            "right": (pybamm.Scalar(0), "Neumann"),
        }
        self.initial_conditions[phi_e] = -param.U_n(param.c_e_init, param.T_init)

        ######################
        # Current in the solid
        ######################
        i_s_n = -param.sigma_n * (1 - eps_n) ** param.b_s_n * pybamm.grad(phi_s_n)
        sigma_eff_p = param.sigma_p * (1 - eps_p) ** param.b_s_p
        i_s_p = -sigma_eff_p * pybamm.grad(phi_s_p)
        # The `algebraic` dictionary contains differential equations, with the key being
        # the main scalar variable of interest in the equation
        self.algebraic[phi_s_n] = pybamm.div(i_s_n) + j_n
        self.algebraic[phi_s_p] = pybamm.div(i_s_p) + j_p
        self.boundary_conditions[phi_s_n] = {
            "left": (pybamm.Scalar(0), "Dirichlet"),
            "right": (pybamm.Scalar(0), "Neumann"),
        }
        self.boundary_conditions[phi_s_p] = {
            "left": (pybamm.Scalar(0), "Neumann"),
            "right": (i_cell / pybamm.boundary_value(-sigma_eff_p, "right"), "Neumann"),
        }
        # Initial conditions must also be provided for algebraic equations, as an
        # initial guess for a root-finding algorithm which calculates consistent initial
        # conditions
        self.initial_conditions[phi_s_n] = pybamm.Scalar(0)
        self.initial_conditions[phi_s_p] = param.U_p(
            param.c_e_init, param.T_init
        ) - param.U_n(param.c_e_init, param.T_init)

        ######################
        # Porosity
        ######################
        beta_surf = pybamm.Concatenation(
            pybamm.PrimaryBroadcast(param.beta_surf_n, "negative electrode"),
            pybamm.PrimaryBroadcast(0, "separator"),
            pybamm.PrimaryBroadcast(param.beta_surf_p, "positive electrode"),
        )
        deps_dt = -beta_surf * j
        self.rhs[eps] = deps_dt
        self.initial_conditions[eps] = param.epsilon_init
        self.events.extend(
            [
                pybamm.Event(
                    "Zero negative electrode porosity cut-off", pybamm.min(eps_n)
                ),
                pybamm.Event(
                    "Max negative electrode porosity cut-off", pybamm.max(eps_n) - 1
                ),
                pybamm.Event(
                    "Zero positive electrode porosity cut-off", pybamm.min(eps_p)
                ),
                pybamm.Event(
                    "Max positive electrode porosity cut-off", pybamm.max(eps_p) - 1
                ),
            ]
        )

        ######################
        # Electrolyte concentration
        ######################
        N_e = (
            -tor * param.D_e(c_e, T) * pybamm.grad(c_e)
            + param.C_e * param.t_plus(c_e) * i_e / param.gamma_e
            + param.C_e * c_e * v
        )
        s = pybamm.Concatenation(
            pybamm.PrimaryBroadcast(param.s_plus_n_S, "negative electrode"),
            pybamm.PrimaryBroadcast(0, "separator"),
            pybamm.PrimaryBroadcast(param.s_plus_p_S, "positive electrode"),
        )
        self.rhs[c_e] = (1 / eps) * (
            -pybamm.div(N_e) / param.C_e
            + s * j / param.gamma_e
            - c_e * deps_dt
            - c_e * div_V
        )
        self.boundary_conditions[c_e] = {
            "left": (pybamm.Scalar(0), "Neumann"),
            "right": (pybamm.Scalar(0), "Neumann"),
        }
        self.initial_conditions[c_e] = param.c_e_init
        self.events.append(
            pybamm.Event(
                "Zero electrolyte concentration cut-off", pybamm.min(c_e) - 0.002
            )
        )

        ######################
        # (Some) variables
        ######################
        voltage = pybamm.boundary_value(phi_s_p, "right")
        # The `variables` dictionary contains all variables that might be useful for
        # visualising the solution of the model
        pot = param.potential_scale

        self.variables = {
            "Electrolyte concentration": c_e,
            "Current [A]": I,
            "Negative electrode potential [V]": pot * phi_s_n,
            "Electrolyte potential [V]": -param.U_n_ref + pot * phi_e,
            "Positive electrode potential [V]": param.U_p_ref
            - param.U_n_ref
            + pot * phi_s_p,
            "Terminal voltage [V]": param.U_p_ref - param.U_n_ref + pot * voltage,
            "x [m]": pybamm.standard_spatial_vars.x * param.L_x,
            "x": pybamm.standard_spatial_vars.x,
            "Porosity": eps,
            "Volume-averaged velocity": v,
            "X-averaged separator transverse volume-averaged velocity": div_V_s,
        }
        self.events.extend(
            [
                pybamm.Event("Minimum voltage", voltage - param.voltage_low_cut),
                pybamm.Event("Maximum voltage", voltage - param.voltage_high_cut),
            ]
        )
