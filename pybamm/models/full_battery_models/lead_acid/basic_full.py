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

    def __init__(self, name="Full model"):
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
            "Negative electrolyte pressure",
            domain="negative electrode",
            auxiliary_domains={"secondary": "current collector"},
        )
        pressure_s = pybamm.Variable(
            "Separator electrolyte pressure",
            domain="separator",
            auxiliary_domains={"secondary": "current collector"},
        )
        pressure_p = pybamm.Variable(
            "Positive electrolyte pressure",
            domain="positive electrode",
            auxiliary_domains={"secondary": "current collector"},
        )
        pressure = pybamm.Concatenation(pressure_n, pressure_s, pressure_p)

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
        j0_n = param.j0_n_S_ref * c_e_n
        j_n = (
            2
            * j0_n
            * pybamm.sinh(param.ne_n / 2 * (phi_s_n - phi_e_n - param.U_n(c_e_n, T)))
        )
        j0_p = param.j0_p_S_ref * c_e_p ** 2 * param.c_w(c_e_p)
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
        v = -pybamm.grad(pressure)
        l_s = pybamm.geometric_parameters.l_s

        # Difference in negative and positive electrode velocities determines the
        # velocity in the separator
        v_box_n_right = param.beta_n * i_cell
        v_box_p_left = param.beta_p * i_cell
        d_vbox_s__dx = (v_box_p_left - v_box_n_right) / l_s

        # Simple formula for velocity in the separator
        dVbox_dz = pybamm.Concatenation(
            pybamm.PrimaryBroadcast(0, "negative electrode"),
            pybamm.PrimaryBroadcast(-d_vbox_s__dx, "separator"),
            pybamm.PrimaryBroadcast(0, "positive electrode"),
        )
        self.algebraic[pressure] = pybamm.div(v) + dVbox_dz - param.beta * j
        self.boundary_conditions[pressure] = {
            "left": (pybamm.Scalar(0), "Dirichlet"),
            "right": (pybamm.Scalar(0), "Neumann"),
        }
        self.initial_conditions[pressure] = pybamm.Scalar(0)

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
        deps_dt = -param.beta_surf * j
        self.rhs[eps] = deps_dt
        self.initial_conditions[eps] = param.epsilon_init
        self.events["Zero negative electrode porosity cut-off"] = pybamm.min(eps_n)
        self.events["Max negative electrode porosity cut-off"] = pybamm.max(eps_n) - 1
        self.events["Zero positive electrode porosity cut-off"] = pybamm.min(eps_p)
        self.events["Max positive electrode porosity cut-off"] = pybamm.max(eps_p) - 1

        ######################
        # Electrolyte concentration
        ######################
        N_e = -tor * param.D_e(c_e, T) * pybamm.grad(c_e) + c_e * v
        self.rhs[c_e] = (1 / eps) * (
            -pybamm.div(N_e) / param.C_e + param.s * j / param.gamma_e - c_e * deps_dt
        )
        self.boundary_conditions[c_e] = {
            "left": (pybamm.Scalar(0), "Neumann"),
            "right": (pybamm.Scalar(0), "Neumann"),
        }
        self.initial_conditions[c_e] = param.c_e_init
        self.events["Zero electrolyte concentration cut-off"] = pybamm.min(c_e) - 0.002

        ######################
        # (Some) variables
        ######################
        voltage = pybamm.boundary_value(phi_s_p, "right")
        # The `variables` dictionary contains all variables that might be useful for
        # visualising the solution of the model
        self.variables = {
            "Electrolyte concentration": c_e,
            "Current [A]": I,
            "Negative electrode potential": phi_s_n,
            "Electrolyte potential": phi_e,
            "Positive electrode potential": phi_s_p,
            "Terminal voltage": voltage,
        }
        self.events["Minimum voltage"] = voltage - param.voltage_low_cut
        self.events["Maximum voltage"] = voltage - param.voltage_high_cut

    @property
    def default_geometry(self):
        return pybamm.Geometry("1D macro")
