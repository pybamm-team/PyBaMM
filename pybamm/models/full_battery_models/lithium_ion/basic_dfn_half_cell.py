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
    feature under development (for example, it cannot be used with the Simulation class
    for the moment) and in the future it will be incorporated as a standard model with
    the full functionality.

    Parameters
    ----------
    name : str, optional
        The name of the model.
    options : dict
        A dictionary of options to be passed to the model. For the half cell it should
        include which is the working electrode.

    References
    ----------
    .. [2] M Doyle, TF Fuller and JS Nwman. “Modeling of Galvanostatic Charge and
        Discharge of the Lithium/Polymer/Insertion Cell”. Journal of The
        Electrochemical Society, 140(6):1526-1533, 1993

    **Extends:** :class:`pybamm.lithium_ion.BaseModel`
    """

    def __init__(
        self, name="Doyle-Fuller-Newman half cell model", options=None,
    ):
        super().__init__({}, name)
        pybamm.citations.register("marquis2019asymptotic")
        # `param` is a class containing all the relevant parameters and functions for
        # this model. These are purely symbolic at this stage, and will be set by the
        # `ParameterValues` class when the model is processed.
        param = self.param
        options = options or {"working electrode": None}

        if options["working electrode"] not in ["negative", "positive"]:
            raise ValueError(
                "The option 'working electrode' should be either 'positive'"
                " or 'negative'"
            )

        self.options.update(options)
        working_electrode = options["working electrode"]

        ######################
        # Variables
        ######################
        # Variables that depend on time only are created without a domain
        Q = pybamm.Variable("Discharge capacity [A.h]")

        # Define some useful scalings
        pot = param.potential_scale
        i_typ = param.current_scale

        # Variables that vary spatially are created with a domain. Depending on
        # which is the working electrode we need to define a set variables or another
        if working_electrode == "negative":
            # Electrolyte concentration
            c_e_n = pybamm.Variable(
                "Negative electrolyte concentration", domain="negative electrode"
            )
            c_e_s = pybamm.Variable(
                "Separator electrolyte concentration", domain="separator"
            )
            # Concatenations combine several variables into a single variable, to
            # simplify implementing equations that hold over several domains
            c_e = pybamm.Concatenation(c_e_n, c_e_s)

            # Electrolyte potential
            phi_e_n = pybamm.Variable(
                "Negative electrolyte potential", domain="negative electrode"
            )
            phi_e_s = pybamm.Variable(
                "Separator electrolyte potential", domain="separator"
            )
            phi_e = pybamm.Concatenation(phi_e_n, phi_e_s)

            # Particle concentrations are variables on the particle domain, but also
            # vary in the x-direction (electrode domain) and so must be provided with
            # auxiliary domains
            c_s_n = pybamm.Variable(
                "Negative particle concentration",
                domain="negative particle",
                auxiliary_domains={"secondary": "negative electrode"},
            )
            # Set concentration in positive particle to be equal to the initial
            # concentration as it is not the working electrode
            x_p = pybamm.PrimaryBroadcast(
                pybamm.standard_spatial_vars.x_p, "positive particle"
            )
            c_s_p = param.c_n_init(x_p)

            # Electrode potential
            phi_s_n = pybamm.Variable(
                "Negative electrode potential", domain="negative electrode"
            )
            # Set potential in positive electrode to be equal to the initial OCV
            phi_s_p = param.U_p(pybamm.surf(param.c_p_init(x_p)), param.T_init)
        else:
            c_e_p = pybamm.Variable(
                "Positive electrolyte concentration", domain="positive electrode"
            )
            c_e_s = pybamm.Variable(
                "Separator electrolyte concentration", domain="separator"
            )
            # Concatenations combine several variables into a single variable, to
            # simplify implementing equations that hold over several domains
            c_e = pybamm.Concatenation(c_e_s, c_e_p)

            # Electrolyte potential
            phi_e_s = pybamm.Variable(
                "Separator electrolyte potential", domain="separator"
            )
            phi_e_p = pybamm.Variable(
                "Positive electrolyte potential", domain="positive electrode"
            )
            phi_e = pybamm.Concatenation(phi_e_s, phi_e_p)

            # Particle concentrations are variables on the particle domain, but also
            # vary in the x-direction (electrode domain) and so must be provided with
            # auxiliary domains
            c_s_p = pybamm.Variable(
                "Positive particle concentration",
                domain="positive particle",
                auxiliary_domains={"secondary": "positive electrode"},
            )
            # Set concentration in negative particle to be equal to the initial
            # concentration as it is not the working electrode
            x_n = pybamm.PrimaryBroadcast(
                pybamm.standard_spatial_vars.x_n, "negative particle"
            )
            c_s_n = param.c_n_init(x_n)

            # Electrode potential
            phi_s_p = pybamm.Variable(
                "Positive electrode potential", domain="positive electrode"
            )
            # Set potential in negative electrode to be equal to the initial OCV
            phi_s_n = param.U_n(pybamm.surf(param.c_n_init(x_n)), param.T_init)

        # Constant temperature
        T = param.T_init

        ######################
        # Other set-up
        ######################

        # Current density
        i_cell = param.current_with_time

        # Porosity and Tortuosity
        # Primary broadcasts are used to broadcast scalar quantities across a domain
        # into a vector of the right shape, for multiplying with other vectors
        eps_n = pybamm.PrimaryBroadcast(
            pybamm.Parameter("Negative electrode porosity"), "negative electrode"
        )
        eps_s = pybamm.PrimaryBroadcast(
            pybamm.Parameter("Separator porosity"), "separator"
        )
        eps_p = pybamm.PrimaryBroadcast(
            pybamm.Parameter("Positive electrode porosity"), "positive electrode"
        )

        if working_electrode == "negative":
            eps = pybamm.Concatenation(eps_n, eps_s)
            tor = pybamm.Concatenation(eps_n ** param.b_e_n, eps_s ** param.b_e_s)
        else:
            eps = pybamm.Concatenation(eps_s, eps_p)
            tor = pybamm.Concatenation(eps_s ** param.b_e_s, eps_p ** param.b_e_p)

        # Interfacial reactions
        # Surf takes the surface value of a variable, i.e. its boundary value on the
        # right side. This is also accessible via `boundary_value(x, "right")`, with
        # "left" providing the boundary value of the left side
        c_s_surf_n = pybamm.surf(c_s_n)
        c_s_surf_p = pybamm.surf(c_s_p)

        if working_electrode == "negative":
            j0_n = param.j0_n(c_e_n, c_s_surf_n, T) / param.C_r_n
            j_n = (
                2
                * j0_n
                * pybamm.sinh(
                    param.ne_n / 2 * (phi_s_n - phi_e_n - param.U_n(c_s_surf_n, T))
                )
            )
            j_s = pybamm.PrimaryBroadcast(0, "separator")
            j_p = pybamm.PrimaryBroadcast(0, "positive electrode")
            j = pybamm.Concatenation(j_n, j_s)
        else:
            j0_p = param.gamma_p * param.j0_p(c_e_p, c_s_surf_p, T) / param.C_r_p
            j_p = (
                2
                * j0_p
                * pybamm.sinh(
                    param.ne_p / 2 * (phi_s_p - phi_e_p - param.U_p(c_s_surf_p, T))
                )
            )
            j_s = pybamm.PrimaryBroadcast(0, "separator")
            j_n = pybamm.PrimaryBroadcast(0, "negative electrode")
            j = pybamm.Concatenation(j_s, j_p)

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
        # Particles
        ######################

        if working_electrode == "negative":
            # The div and grad operators will be converted to the appropriate matrix
            # multiplication at the discretisation stage
            N_s_n = -param.D_n(c_s_n, T) * pybamm.grad(c_s_n)
            self.rhs[c_s_n] = -(1 / param.C_n) * pybamm.div(N_s_n)

            # Boundary conditions must be provided for equations with spatial
            # derivatives
            self.boundary_conditions[c_s_n] = {
                "left": (pybamm.Scalar(0), "Neumann"),
                "right": (
                    -param.C_n * j_n / param.a_n / param.D_n(c_s_surf_n, T),
                    "Neumann",
                ),
            }

            # c_n_init can in general be a function of x
            # Note the broadcasting, for domains
            x_n = pybamm.PrimaryBroadcast(
                pybamm.standard_spatial_vars.x_n, "negative particle"
            )
            self.initial_conditions[c_s_n] = param.c_n_init(x_n)

            # Events specify points at which a solution should terminate
            self.events += [
                pybamm.Event(
                    "Minimum negative particle surface concentration",
                    pybamm.min(c_s_surf_n) - 0.01,
                ),
                pybamm.Event(
                    "Maximum negative particle surface concentration",
                    (1 - 0.01) - pybamm.max(c_s_surf_n),
                ),
            ]
        else:
            # The div and grad operators will be converted to the appropriate matrix
            # multiplication at the discretisation stage
            N_s_p = -param.D_p(c_s_p, T) * pybamm.grad(c_s_p)
            self.rhs[c_s_p] = -(1 / param.C_p) * pybamm.div(N_s_p)

            # Boundary conditions must be provided for equations with spatial
            # derivatives
            self.boundary_conditions[c_s_p] = {
                "left": (pybamm.Scalar(0), "Neumann"),
                "right": (
                    -param.C_p
                    * j_p
                    / param.a_p
                    / param.gamma_p
                    / param.D_p(c_s_surf_p, T),
                    "Neumann",
                ),
            }

            # c_p_init can in general be a function of x
            # Note the broadcasting, for domains
            x_p = pybamm.PrimaryBroadcast(
                pybamm.standard_spatial_vars.x_p, "positive particle"
            )
            self.initial_conditions[c_s_p] = param.c_p_init(x_p)

            # Events specify points at which a solution should terminate
            self.events += [
                pybamm.Event(
                    "Minimum positive particle surface concentration",
                    pybamm.min(c_s_surf_p) - 0.01,
                ),
                pybamm.Event(
                    "Maximum positive particle surface concentration",
                    (1 - 0.01) - pybamm.max(c_s_surf_p),
                ),
            ]

        ######################
        # Current in the solid
        ######################
        eps_s_n = pybamm.Parameter("Negative electrode active material volume fraction")
        eps_s_p = pybamm.Parameter("Positive electrode active material volume fraction")

        if working_electrode == "negative":
            sigma_eff_n = param.sigma_n * eps_s_n ** param.b_s_n
            i_s_n = -sigma_eff_n * pybamm.grad(phi_s_n)
            self.boundary_conditions[phi_s_n] = {
                "left": (
                    i_cell / pybamm.boundary_value(-sigma_eff_n, "left"),
                    "Neumann",
                ),
                "right": (pybamm.Scalar(0), "Neumann"),
            }
            # The `algebraic` dictionary contains differential equations, with the key
            # being the main scalar variable of interest in the equation
            self.algebraic[phi_s_n] = pybamm.div(i_s_n) + j_n

            # Initial conditions must also be provided for algebraic equations, as an
            # initial guess for a root-finding algorithm which calculates consistent
            # initial conditions
            self.initial_conditions[phi_s_n] = param.U_n(
                param.c_n_init(0), param.T_init
            )
        else:
            sigma_eff_p = param.sigma_p * eps_s_p ** param.b_s_p
            i_s_p = -sigma_eff_p * pybamm.grad(phi_s_p)
            self.boundary_conditions[phi_s_p] = {
                "left": (pybamm.Scalar(0), "Neumann"),
                "right": (
                    i_cell / pybamm.boundary_value(-sigma_eff_p, "right"),
                    "Neumann",
                ),
            }
            self.algebraic[phi_s_p] = pybamm.div(i_s_p) + j_p
            # Initial conditions must also be provided for algebraic equations, as an
            # initial guess for a root-finding algorithm which calculates consistent
            # initial conditions
            self.initial_conditions[phi_s_p] = param.U_p(
                param.c_p_init(1), param.T_init
            )

        ######################
        # Electrolyte concentration
        ######################
        N_e = -tor * param.D_e(c_e, T) * pybamm.grad(c_e)
        self.rhs[c_e] = (1 / eps) * (
            -pybamm.div(N_e) / param.C_e + (1 - param.t_plus(c_e)) * j / param.gamma_e
        )
        dce_dx = (
            -(1 - param.t_plus(c_e))
            * i_cell
            * param.C_e
            / (tor * param.gamma_e * param.D_e(c_e, T))
        )

        if working_electrode == "negative":
            self.boundary_conditions[c_e] = {
                "left": (pybamm.Scalar(0), "Neumann"),
                "right": (pybamm.boundary_value(dce_dx, "right"), "Neumann"),
            }
        else:
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
        i_e = (param.kappa_e(c_e, T) * tor * param.gamma_e / param.C_e) * (
            param.chi(c_e) * pybamm.grad(c_e) / c_e - pybamm.grad(phi_e)
        )
        self.algebraic[phi_e] = pybamm.div(i_e) - j

        ref_potential = param.U_n_ref / pot

        if working_electrode == "negative":
            self.boundary_conditions[phi_e] = {
                "left": (pybamm.Scalar(0), "Neumann"),
                "right": (ref_potential, "Dirichlet"),
            }
        else:
            self.boundary_conditions[phi_e] = {
                "left": (ref_potential, "Dirichlet"),
                "right": (pybamm.Scalar(0), "Neumann"),
            }

        self.initial_conditions[phi_e] = ref_potential
        ######################
        # (Some) variables
        ######################
        L_Li = pybamm.Parameter("Lithium counter electrode thickness [m]")
        sigma_Li = pybamm.Parameter("Lithium counter electrode conductivity [S.m-1]")
        j_Li = pybamm.Parameter(
            "Lithium counter electrode exchange-current density [A.m-2]"
        )

        if working_electrode == "negative":
            voltage = pybamm.boundary_value(phi_s_n, "left") - ref_potential
            voltage_dim = pot * pybamm.boundary_value(phi_s_n, "left")
            vdrop_Li = 2 * pybamm.arcsinh(
                i_cell * i_typ / j_Li
            ) + L_Li * i_typ * i_cell / (sigma_Li * pot)
            vdrop_Li_dim = (
                2 * pot * pybamm.arcsinh(i_cell * i_typ / j_Li)
                + L_Li * i_typ * i_cell / sigma_Li
            )
        else:
            voltage = pybamm.boundary_value(phi_s_p, "right") - ref_potential
            voltage_dim = param.U_p_ref + pot * voltage
            vdrop_Li = -(
                2 * pybamm.arcsinh(i_cell * i_typ / j_Li)
                + L_Li * i_typ * i_cell / (sigma_Li * pot)
            )
            vdrop_Li_dim = -(
                2 * pot * pybamm.arcsinh(i_cell * i_typ / j_Li)
                + L_Li * i_typ * i_cell / sigma_Li
            )

        c_s_surf_p_av = pybamm.x_average(c_s_surf_p)
        c_s_surf_n_av = pybamm.x_average(c_s_surf_n)

        # The `variables` dictionary contains all variables that might be useful for
        # visualising the solution of the model
        self.variables = {
            "Time [s]": param.timescale * pybamm.t,
            "Negative particle surface concentration": c_s_surf_n,
            "X-averaged negative particle surface concentration": c_s_surf_n_av,
            "Negative particle concentration": c_s_n,
            "Negative particle surface concentration [mol.m-3]": param.c_n_max
            * c_s_surf_n,
            "X-averaged negative particle surface concentration [mol.m-3]":
            param.c_n_max * c_s_surf_n_av,
            "Negative particle concentration [mol.m-3]": param.c_n_max * c_s_n,
            "Electrolyte concentration": c_e,
            "Electrolyte concentration [mol.m-3]": param.c_e_typ * c_e,
            "Positive particle surface concentration": c_s_surf_p,
            "X-averaged positive particle surface concentration": c_s_surf_p_av,
            "Positive particle concentration": c_s_p,
            "Positive particle surface concentration [mol.m-3]": param.c_p_max
            * c_s_surf_p,
            "X-averaged positive particle surface concentration [mol.m-3]":
            param.c_p_max * c_s_surf_p_av,
            "Positive particle concentration [mol.m-3]": param.c_p_max * c_s_p,
            "Current [A]": I,
            "Negative electrode potential": phi_s_n,
            "Negative electrode potential [V]": pot * phi_s_n,
            "Negative electrode open circuit potential": param.U_n(c_s_surf_n, T),
            "Electrolyte potential": phi_e,
            "Electrolyte potential [V]": -param.U_n_ref + pot * phi_e,
            "Positive electrode potential": phi_s_p,
            "Positive electrode potential [V]": (param.U_p_ref - param.U_n_ref)
            + pot * phi_s_p,
            "Positive electrode open circuit potential": param.U_p(c_s_surf_p, T),
            "Voltage drop": voltage,
            "Voltage drop [V]": voltage_dim,
            "Terminal voltage": voltage + vdrop_Li,
            "Terminal voltage [V]": voltage_dim + vdrop_Li_dim,
        }

    def new_copy(self, build=False):
        new_model = self.__class__(name=self.name, options=self.options)
        new_model.use_jacobian = self.use_jacobian
        new_model.use_simplify = self.use_simplify
        new_model.convert_to_format = self.convert_to_format
        new_model.timescale = self.timescale
        new_model.length_scales = self.length_scales
        return new_model
