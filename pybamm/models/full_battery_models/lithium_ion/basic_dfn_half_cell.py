#
# Basic Doyle-Fuller-Newman (DFN) Half Cell Model
#
import pybamm
from .base_lithium_ion_model import BaseModel


class BasicDFNHalfCell(BaseModel):
    """Doyle-Fuller-Newman (DFN) model of a lithium-ion battery with lithium counter
    electrode, adapted from [2]_.

    This class differs from the :class:`pybamm.lithium_ion.BasicDFN` model class in
    that it is for a cell with a lithium counter electrode (half cell). This is a
    feature under development (for example, it cannot be used with the Experiment class
    for the moment) and in the future it will be incorporated as a standard model with
    the full functionality.

    The electrode labeled "positive electrode" is the working electrode, and the
    electrode labeled "negative electrode" is the counter electrode. If the "negative
    electrode" is the working electrode, then the parameters for the "negative
    electrode" are used to define the "positive electrode".
    This facilitates compatibility with the full-cell models.

    Parameters
    ----------
    options : dict
        A dictionary of options to be passed to the model. For the half cell it should
        include which is the working electrode.
    name : str, optional
        The name of the model.

    References
    ----------
    .. [2] M Doyle, TF Fuller and JS Nwman. “Modeling of Galvanostatic Charge and
        Discharge of the Lithium/Polymer/Insertion Cell”. Journal of The
        Electrochemical Society, 140(6):1526-1533, 1993

    **Extends:** :class:`pybamm.lithium_ion.BaseModel`
    """

    def __init__(self, options=None, name="Doyle-Fuller-Newman half cell model"):
        super().__init__(options, name)
        if self.options["working electrode"] not in ["negative", "positive"]:
            raise ValueError(
                "The option 'working electrode' should be either 'positive'"
                " or 'negative'"
            )
        pybamm.citations.register("Marquis2019")
        # `param` is a class containing all the relevant parameters and functions for
        # this model. These are purely symbolic at this stage, and will be set by the
        # `ParameterValues` class when the model is processed.
        param = self.param

        R_w_typ = param.p.prim.R_typ

        # Set default length scales
        self._length_scales = {
            "separator": param.L_x,
            "positive electrode": param.L_x,
            "positive particle": R_w_typ,
            "current collector y": param.L_z,
            "current collector z": param.L_z,
        }

        ######################
        # Variables
        ######################
        # Variables that depend on time only are created without a domain
        Q = pybamm.Variable("Discharge capacity [A.h]")

        # Define some useful scalings
        pot_scale = param.potential_scale
        i_typ = param.current_scale

        # Variables that vary spatially are created with a domain.
        c_e_s = pybamm.Variable(
            "Separator electrolyte concentration", domain="separator"
        )
        c_e_w = pybamm.Variable(
            "Positive electrolyte concentration", domain="positive electrode"
        )
        c_e = pybamm.concatenation(c_e_s, c_e_w)
        c_s_w = pybamm.Variable(
            "Positive particle concentration",
            domain="positive particle",
            auxiliary_domains={"secondary": "positive electrode"},
        )
        phi_s_w = pybamm.Variable(
            "Positive electrode potential", domain="positive electrode"
        )
        phi_e_s = pybamm.Variable("Separator electrolyte potential", domain="separator")
        phi_e_w = pybamm.Variable(
            "Positive electrolyte potential", domain="positive electrode"
        )
        phi_e = pybamm.concatenation(phi_e_s, phi_e_w)

        # Constant temperature
        T = param.T_init

        ######################
        # Other set-up
        ######################

        # Current density
        i_cell = param.current_with_time

        # Define particle surface concentration
        # Surf takes the surface value of a variable, i.e. its boundary value on the
        # right side. This is also accessible via `boundary_value(x, "right")`, with
        # "left" providing the boundary value of the left side
        c_s_surf_w = pybamm.surf(c_s_w)

        # Define parameters. We need to assemble them differently depending on the
        # working electrode

        # Porosity and Transport_efficiency
        eps_s = pybamm.PrimaryBroadcast(
            pybamm.Parameter("Separator porosity"), "separator"
        )
        eps_w = pybamm.PrimaryBroadcast(
            pybamm.Parameter("Positive electrode porosity"), "positive electrode"
        )
        b_e_s = param.s.b_e
        b_e_w = param.p.b_e

        # Interfacial reactions
        j0_w = param.p.prim.j0(c_e_w, c_s_surf_w, T)
        U_w = param.p.prim.U
        ne_w = param.p.prim.ne

        # Particle diffusion parameters
        D_w = param.p.prim.D
        Q_w = param.p.prim.Q_init
        a_R_w = param.p.prim.a_R
        gamma_e = param.c_e_typ / param.p.prim.c_max
        c_w_init = param.p.prim.c_init

        # Electrode equation parameters
        eps_s_w = pybamm.Parameter("Positive electrode active material volume fraction")
        b_s_w = param.p.b_s
        sigma_w = param.p.sigma

        # Other parameters (for outputs)
        c_w_max = param.p.prim.c_max
        U_w_ref = param.p.U_ref
        U_Li_ref = param.n.U_ref
        L_w = param.p.L

        # gamma_w is always 1 because we choose the timescale based on the working
        # electrode
        gamma_w = pybamm.Scalar(1)

        eps = pybamm.concatenation(eps_s, eps_w)
        tor = pybamm.concatenation(eps_s**b_e_s, eps_w**b_e_w)

        j_w = (
            2 * j0_w * pybamm.sinh(ne_w / 2 * (phi_s_w - phi_e_w - U_w(c_s_surf_w, T)))
        )
        j_s = pybamm.PrimaryBroadcast(0, "separator")
        j = pybamm.concatenation(j_s, j_w)

        ######################
        # State of Charge
        ######################
        I = param.dimensional_current_with_time
        # The `rhs` dictionary contains differential equations, with the key being the
        # variable in the d/dt
        self.rhs[Q] = I * self.timescale / 3600
        # Initial conditions must be provided for the ODEs
        self.initial_conditions[Q] = pybamm.Scalar(0)

        ######################
        # Particles
        ######################
        # The div and grad operators will be converted to the appropriate matrix
        # multiplication at the discretisation stage
        N_s_w = -D_w(c_s_w, T) * pybamm.grad(c_s_w)
        self.rhs[c_s_w] = -(1 / Q_w) * pybamm.div(N_s_w)

        # Boundary conditions must be provided for equations with spatial
        # derivatives
        self.boundary_conditions[c_s_w] = {
            "left": (pybamm.Scalar(0), "Neumann"),
            "right": (
                -Q_w * j_w / a_R_w / gamma_w / D_w(c_s_surf_w, T),
                "Neumann",
            ),
        }
        self.initial_conditions[c_s_w] = c_w_init

        # Events specify points at which a solution should terminate
        self.events += [
            pybamm.Event(
                "Minimum positive particle surface concentration",
                pybamm.min(c_s_surf_w) - 0.01,
            ),
            pybamm.Event(
                "Maximum positive particle surface concentration",
                (1 - 0.01) - pybamm.max(c_s_surf_w),
            ),
        ]

        ######################
        # Current in the solid
        ######################
        sigma_eff_w = sigma_w(T) * eps_s_w**b_s_w
        i_s_w = -sigma_eff_w * pybamm.grad(phi_s_w)
        self.boundary_conditions[phi_s_w] = {
            "left": (pybamm.Scalar(0), "Neumann"),
            "right": (
                i_cell / pybamm.boundary_value(-sigma_eff_w, "right"),
                "Neumann",
            ),
        }
        self.algebraic[phi_s_w] = pybamm.div(i_s_w) + j_w
        # Initial conditions must also be provided for algebraic equations, as an
        # initial guess for a root-finding algorithm which calculates consistent
        # initial conditions
        self.initial_conditions[phi_s_w] = param.p.prim.U_init

        ######################
        # Electrolyte concentration
        ######################
        N_e = -tor * param.D_e(c_e, T) * pybamm.grad(c_e)
        self.rhs[c_e] = (1 / eps) * (
            -pybamm.div(N_e) / param.C_e + (1 - param.t_plus(c_e, T)) * j / gamma_e
        )
        dce_dx = (
            -(1 - param.t_plus(c_e, T))
            * i_cell
            * param.C_e
            / (tor * gamma_e * param.D_e(c_e, T))
        )

        self.boundary_conditions[c_e] = {
            "left": (pybamm.boundary_value(dce_dx, "left"), "Neumann"),
            "right": (pybamm.Scalar(0), "Neumann"),
        }

        self.initial_conditions[c_e] = param.c_e_init
        self.events.append(
            pybamm.Event(
                "Zero electrolyte concentration cut-off", pybamm.min(c_e) - 0.002
            )
        )

        ######################
        # Current in the electrolyte
        ######################
        i_e = (param.kappa_e(c_e, T) * tor * gamma_e / param.C_e) * (
            param.chiRT_over_Fc(c_e, T) * pybamm.grad(c_e) - pybamm.grad(phi_e)
        )
        self.algebraic[phi_e] = pybamm.div(i_e) - j

        # dimensionless reference potential so that dimensional reference potential
        # is zero (phi_dim = U_n_ref + pot_scale * phi)
        l_Li = param.p.l
        sigma_Li = param.p.sigma
        j_Li = param.j0_plating(pybamm.boundary_value(c_e, "left"), 1, T)
        eta_Li = 2 * (1 + param.Theta * T) * pybamm.arcsinh(i_cell / (2 * j_Li))

        phi_s_cn = 0
        delta_phi = eta_Li + U_Li_ref
        delta_phis_Li = l_Li * i_cell / sigma_Li(T)
        ref_potential = phi_s_cn - delta_phis_Li - delta_phi

        self.boundary_conditions[phi_e] = {
            "left": (ref_potential, "Dirichlet"),
            "right": (pybamm.Scalar(0), "Neumann"),
        }

        self.initial_conditions[phi_e] = param.n.U_ref / pot_scale

        ######################
        # (Some) variables
        ######################
        vdrop_cell = pybamm.boundary_value(phi_s_w, "right") - ref_potential
        vdrop_Li = -eta_Li - delta_phis_Li
        voltage = vdrop_cell + vdrop_Li
        voltage_dim = U_w_ref - U_Li_ref + pot_scale * voltage
        c_e_total = pybamm.x_average(eps * c_e)
        c_s_surf_w_av = pybamm.x_average(c_s_surf_w)

        c_s_rav = pybamm.r_average(c_s_w)
        c_s_vol_av = pybamm.x_average(eps_s_w * c_s_rav)

        # Cut-off voltage
        self.events.append(
            pybamm.Event(
                "Minimum voltage",
                voltage_dim - self.param.voltage_low_cut_dimensional,
                pybamm.EventType.TERMINATION,
            )
        )
        self.events.append(
            pybamm.Event(
                "Maximum voltage",
                self.param.voltage_high_cut_dimensional - voltage_dim,
                pybamm.EventType.TERMINATION,
            )
        )

        # Cut-off open-circuit voltage (for event switch with casadi 'fast with events'
        # mode)
        tol = 0.1
        self.events.append(
            pybamm.Event(
                "Minimum voltage switch",
                voltage_dim - (self.param.voltage_low_cut_dimensional - tol),
                pybamm.EventType.SWITCH,
            )
        )
        self.events.append(
            pybamm.Event(
                "Maximum voltage switch",
                voltage_dim - (self.param.voltage_high_cut_dimensional + tol),
                pybamm.EventType.SWITCH,
            )
        )

        # The `variables` dictionary contains all variables that might be useful for
        # visualising the solution of the model
        self.variables = {
            "Time [s]": self.timescale * pybamm.t,
            "Positive particle surface concentration": c_s_surf_w,
            "X-averaged positive particle surface concentration": c_s_surf_w_av,
            "Positive particle concentration": c_s_w,
            "Positive particle surface concentration [mol.m-3]": c_w_max * c_s_surf_w,
            "X-averaged positive particle surface concentration "
            "[mol.m-3]": c_w_max * c_s_surf_w_av,
            "Positive particle concentration [mol.m-3]": c_w_max * c_s_w,
            "Total lithium in positive electrode": c_s_vol_av,
            "Total lithium in positive electrode [mol]": c_s_vol_av
            * c_w_max
            * L_w
            * param.A_cc,
            "Electrolyte concentration": c_e,
            "Electrolyte concentration [mol.m-3]": param.c_e_typ * c_e,
            "Total lithium in electrolyte": c_e_total,
            "Total lithium in electrolyte [mol]": c_e_total
            * param.c_e_typ
            * param.L_x
            * param.A_cc,
            "Current [A]": I,
            "Current density [A.m-2]": i_cell * i_typ,
            "Positive electrode potential": phi_s_w,
            "Positive electrode potential [V]": U_w_ref
            - U_Li_ref
            + pot_scale * phi_s_w,
            "Positive electrode open circuit potential": U_w(c_s_surf_w, T),
            "Positive electrode open circuit potential [V]": U_w_ref
            + pot_scale * U_w(c_s_surf_w, T),
            "Electrolyte potential": phi_e,
            "Electrolyte potential [V]": -U_Li_ref + pot_scale * phi_e,
            "Voltage drop in the cell": vdrop_cell,
            "Voltage drop in the cell [V]": U_w_ref - U_Li_ref + pot_scale * vdrop_cell,
            "Negative electrode exchange current density": j_Li,
            "Negative electrode reaction overpotential": eta_Li,
            "Negative electrode reaction overpotential [V]": pot_scale * eta_Li,
            "Negative electrode potential drop": delta_phis_Li,
            "Negative electrode potential drop [V]": pot_scale * delta_phis_Li,
            "Terminal voltage": voltage,
            "Terminal voltage [V]": voltage_dim,
            "Instantaneous power [W.m-2]": i_cell * i_typ * voltage_dim,
            "Pore-wall flux [mol.m-2.s-1]": j_w,
        }

    def new_copy(self, build=False):
        new_model = self.__class__(name=self.name, options=self.options)
        new_model.use_jacobian = self.use_jacobian
        new_model.convert_to_format = self.convert_to_format
        new_model._timescale = self.timescale
        new_model._length_scales = self.length_scales
        return new_model
