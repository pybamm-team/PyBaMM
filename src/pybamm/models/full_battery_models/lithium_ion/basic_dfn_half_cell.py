#
# Basic Doyle-Fuller-Newman (DFN) Half Cell Model
#
import pybamm
from .base_lithium_ion_model import BaseModel


class BasicDFNHalfCell(BaseModel):
    """Doyle-Fuller-Newman (DFN) model of a lithium-ion battery with lithium counter
    electrode, adapted from :footcite:t:`Doyle1993`.

    This class differs from the :class:`pybamm.lithium_ion.BasicDFN` model class in
    that it is for a cell with a lithium counter electrode (half cell). This is a
    feature under development (for example, it cannot be used with the Experiment class
    for the moment) and in the future it will be incorporated as a standard model with
    the full functionality.

    The electrode labeled "positive electrode" is the working electrode, and the
    electrode labeled "negative electrode" is the counter electrode.
    This facilitates compatibility with the full-cell models.

    Parameters
    ----------
    options : dict
        A dictionary of options to be passed to the model. For the half cell it should
        include which is the working electrode.
    name : str, optional
        The name of the model.

    """

    def __init__(self, options=None, name="Doyle-Fuller-Newman half cell model"):
        options = {"working electrode": "positive"}
        super().__init__(options, name)
        pybamm.citations.register("Marquis2019")
        # `param` is a class containing all the relevant parameters and functions for
        # this model. These are purely symbolic at this stage, and will be set by the
        # `ParameterValues` class when the model is processed.

        ######################
        # Variables
        ######################
        # Variables that depend on time only are created without a domain
        Q = pybamm.Variable("Discharge capacity [A.h]")

        # Variables that vary spatially are created with a domain.
        c_e_s = pybamm.Variable(
            "Separator electrolyte concentration [mol.m-3]", domain="separator"
        )
        c_e_w = pybamm.Variable(
            "Positive electrolyte concentration [mol.m-3]", domain="positive electrode"
        )
        c_e = pybamm.concatenation(c_e_s, c_e_w)
        c_s_w = pybamm.Variable(
            "Positive particle concentration [mol.m-3]",
            domain="positive particle",
            auxiliary_domains={"secondary": "positive electrode"},
        )
        phi_s_w = pybamm.Variable(
            "Positive electrode potential [V]", domain="positive electrode"
        )
        phi_e_s = pybamm.Variable(
            "Separator electrolyte potential [V]", domain="separator"
        )
        phi_e_w = pybamm.Variable(
            "Positive electrolyte potential [V]", domain="positive electrode"
        )
        phi_e = pybamm.concatenation(phi_e_s, phi_e_w)

        # Constant temperature
        T = self.param.T_init

        ######################
        # Other set-up
        ######################

        # Current density
        i_cell = self.param.current_density_with_time

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
        b_e_s = self.param.s.b_e
        b_e_w = self.param.p.b_e

        # Interfacial reactions
        j0_w = self.param.p.prim.j0(c_e_w, c_s_surf_w, T)
        U_w = self.param.p.prim.U
        ne_w = self.param.p.prim.ne

        # Particle diffusion parameters
        D_w = self.param.p.prim.D
        c_w_init = self.param.p.prim.c_init

        # Electrode equation parameters
        eps_s_w = pybamm.Parameter("Positive electrode active material volume fraction")
        b_s_w = self.param.p.b_s
        sigma_w = self.param.p.sigma

        # Other parameters (for outputs)
        c_w_max = self.param.p.prim.c_max
        L_w = self.param.p.L

        eps = pybamm.concatenation(eps_s, eps_w)
        tor = pybamm.concatenation(eps_s**b_e_s, eps_w**b_e_w)

        F_RT = self.param.F / (self.param.R * T)
        RT_F = self.param.R * T / self.param.F
        sto_surf_w = c_s_surf_w / c_w_max
        j_w = (
            2
            * j0_w
            * pybamm.sinh(ne_w / 2 * F_RT * (phi_s_w - phi_e_w - U_w(sto_surf_w, T)))
        )
        R_w = self.param.p.prim.R_typ
        a_w = 3 * eps_s_w / R_w
        a_j_w = a_w * j_w
        a_j_s = pybamm.PrimaryBroadcast(0, "separator")
        a_j = pybamm.concatenation(a_j_s, a_j_w)

        ######################
        # State of Charge
        ######################
        I = self.param.current_with_time
        # The `rhs` dictionary contains differential equations, with the key being the
        # variable in the d/dt
        self.rhs[Q] = I / 3600
        # Initial conditions must be provided for the ODEs
        self.initial_conditions[Q] = pybamm.Scalar(0)

        ######################
        # Particles
        ######################
        # The div and grad operators will be converted to the appropriate matrix
        # multiplication at the discretisation stage
        N_s_w = -D_w(c_s_w, T) * pybamm.grad(c_s_w)
        self.rhs[c_s_w] = -pybamm.div(N_s_w)

        # Boundary conditions must be provided for equations with spatial
        # derivatives
        self.boundary_conditions[c_s_w] = {
            "left": (pybamm.Scalar(0), "Neumann"),
            "right": (-j_w / pybamm.surf(D_w(c_s_w, T)) / self.param.F, "Neumann"),
        }
        self.initial_conditions[c_s_w] = c_w_init

        # Events specify points at which a solution should terminate
        self.events += [
            pybamm.Event(
                "Minimum positive particle surface concentration",
                pybamm.min(sto_surf_w) - 0.01,
            ),
            pybamm.Event(
                "Maximum positive particle surface concentration",
                (1 - 0.01) - pybamm.max(sto_surf_w),
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
        # multiply by Lx**2 to improve conditioning
        self.algebraic[phi_s_w] = self.param.L_x**2 * (pybamm.div(i_s_w) + a_j_w)
        # Initial conditions must also be provided for algebraic equations, as an
        # initial guess for a root-finding algorithm which calculates consistent
        # initial conditions
        self.initial_conditions[phi_s_w] = self.param.p.prim.U_init

        ######################
        # Electrolyte concentration
        ######################
        N_e = -tor * self.param.D_e(c_e, T) * pybamm.grad(c_e)
        self.rhs[c_e] = (1 / eps) * (
            -pybamm.div(N_e) + (1 - self.param.t_plus(c_e, T)) * a_j / self.param.F
        )
        dce_dx = (
            -(1 - self.param.t_plus(c_e, T))
            * i_cell
            / (tor * self.param.F * self.param.D_e(c_e, T))
        )

        self.boundary_conditions[c_e] = {
            "left": (pybamm.boundary_value(dce_dx, "left"), "Neumann"),
            "right": (pybamm.Scalar(0), "Neumann"),
        }

        self.initial_conditions[c_e] = self.param.c_e_init
        self.events.append(
            pybamm.Event(
                "Zero electrolyte concentration cut-off", pybamm.min(c_e) - 0.002
            )
        )

        ######################
        # Current in the electrolyte
        ######################
        i_e = (self.param.kappa_e(c_e, T) * tor) * (
            self.param.chiRT_over_Fc(c_e, T) * pybamm.grad(c_e) - pybamm.grad(phi_e)
        )
        # multiply by Lx**2 to improve conditioning
        self.algebraic[phi_e] = self.param.L_x**2 * (pybamm.div(i_e) - a_j)

        # reference potential
        L_Li = self.param.n.L
        sigma_Li = self.param.n.sigma
        j_Li = self.param.j0_Li_metal(pybamm.boundary_value(c_e, "left"), c_w_max, T)
        eta_Li = 2 * RT_F * pybamm.arcsinh(i_cell / (2 * j_Li))

        phi_s_cn = 0
        delta_phi = eta_Li
        delta_phis_Li = L_Li * i_cell / sigma_Li(T)
        ref_potential = phi_s_cn - delta_phis_Li - delta_phi

        self.boundary_conditions[phi_e] = {
            "left": (ref_potential, "Dirichlet"),
            "right": (pybamm.Scalar(0), "Neumann"),
        }

        self.initial_conditions[phi_e] = -self.param.n.prim.U_init

        ######################
        # (Some) variables
        ######################
        vdrop_cell = pybamm.boundary_value(phi_s_w, "right") - ref_potential
        vdrop_Li = -eta_Li - delta_phis_Li
        voltage = vdrop_cell + vdrop_Li
        num_cells = pybamm.Parameter(
            "Number of cells connected in series to make a battery"
        )

        c_e_total = pybamm.x_average(eps * c_e)
        c_s_surf_w_av = pybamm.x_average(c_s_surf_w)

        c_s_rav = pybamm.r_average(c_s_w)
        c_s_vol_av = pybamm.x_average(eps_s_w * c_s_rav)

        # Cut-off voltage
        self.events.append(
            pybamm.Event("Minimum voltage [V]", voltage - self.param.voltage_low_cut)
        )
        self.events.append(
            pybamm.Event("Maximum voltage [V]", self.param.voltage_high_cut - voltage)
        )

        # Cut-off open-circuit voltage (for event switch with casadi 'fast with events'
        # mode)
        tol = 0.1
        self.events.append(
            pybamm.Event(
                "Minimum voltage switch",
                voltage - (self.param.voltage_low_cut - tol),
                pybamm.EventType.SWITCH,
            )
        )
        self.events.append(
            pybamm.Event(
                "Maximum voltage switch",
                voltage - (self.param.voltage_high_cut + tol),
                pybamm.EventType.SWITCH,
            )
        )

        # The `variables` dictionary contains all variables that might be useful for
        # visualising the solution of the model
        self.variables = {
            "Time [s]": pybamm.t,
            "Discharge capacity [A.h]": Q,
            "Positive particle surface concentration [mol.m-3]": c_s_surf_w,
            "X-averaged positive particle surface concentration "
            "[mol.m-3]": c_s_surf_w_av,
            "Positive particle concentration [mol.m-3]": c_s_w,
            "Total lithium in positive electrode [mol]": c_s_vol_av
            * L_w
            * self.param.A_cc,
            "Electrolyte concentration [mol.m-3]": c_e,
            "Separator electrolyte concentration [mol.m-3]": c_e_s,
            "Positive electrolyte concentration [mol.m-3]": c_e_w,
            "Total lithium in electrolyte [mol]": c_e_total
            * self.param.L_x
            * self.param.A_cc,
            "Current [A]": I,
            "Current variable [A]": I,  # for compatibility with pybamm.Experiment
            "Current density [A.m-2]": i_cell,
            "Positive electrode potential [V]": phi_s_w,
            "Positive electrode open-circuit potential [V]": U_w(sto_surf_w, T),
            "Electrolyte potential [V]": phi_e,
            "Separator electrolyte potential [V]": phi_e_s,
            "Positive electrolyte potential [V]": phi_e_w,
            "Voltage drop in the cell [V]": vdrop_cell,
            "Negative electrode exchange current density [A.m-2]": j_Li,
            "Negative electrode reaction overpotential [V]": eta_Li,
            "Negative electrode potential drop [V]": delta_phis_Li,
            "Voltage [V]": voltage,
            "Battery voltage [V]": voltage * num_cells,
            "Instantaneous power [W.m-2]": i_cell * voltage,
        }
