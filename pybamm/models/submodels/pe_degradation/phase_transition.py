#
# Class for the core-shell submodel for PE phase transition.
#
import pybamm
from .base_phase_transition import BasePhaseTransition


class PhaseTransition(BasePhaseTransition):
    """
    Class for positive electrode degradation mechanism of progressive phase
    transition from layer structure to spinel/rocksalt structure,
    especially for NMC811 materials
    aka core-shell model considering
    lithium diffusion in the core,
    oxygen diffusion in the shell, and
    core-shell boundary moving

    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel
    domain : str, optional
        The domain of the model (default is "Positive")
    options: dict
        A dictionary of options to be passed to the model.
        See :class:`pybamm.BaseBatteryModel`
    phase : str, optional
        Phase of the particle (default is "primary")
    x_average : bool
        Whether the particle concentration is averaged over the x-direction
    """

    def __init__(self, param, domain, options, phase="primary", x_average=False):
        # check whether the domain is positive
        if domain != "positive":
            raise DomainError(
                "Domain must be 'positive' for phase transition degradation"
            )

        super().__init__(param, domain, options, phase)
        self.x_average = x_average

        pybamm.citations.register("Zhuo2023")

    def get_fundamental_variables(self):
        domain, Domain = self.domain_Domain
        phase_name = self.phase_name

        variables = {}

        if self.x_average is False:
            # lithium concentration in particle core
            c_c = pybamm.Variable(
                f"{Domain} {phase_name}core lithium concentration [mol.m-3]",
                f"{domain} {phase_name}core",
                auxiliary_domains={
                    "secondary": f"{domain} electrode",
                    "tertiary": "current collector",
                },
                bounds=(0, self.phase_param.c_max),
                scale=self.phase_param.c_max,
            )
            c_c.print_name = f"c_c_{domain[0]}"  # e.g. c_c_n

            # oxygen concentration in degraded shell
            c_o = pybamm.Variable(
                f"{Domain} {phase_name}shell oxygen concentration [mol.m-3]",
                f"{domain} {phase_name}shell",
                auxiliary_domains={
                    "secondary": f"{domain} electrode",
                    "tertiary": "current collector",
                },
                # bounds=(0, self.phase_param.c_max),
                # scale=self.phase_param.c_max,
            )

            # location of the moving core-shell phase boundary, normalized to R
            s_nd = pybamm.Variable(
                f"{Domain} {phase_name}particle moving phase boundary location",
                f"{domain} electrode",
                auxiliary_domains={
                    "secondary": "current collector",
                },
                bounds=(0, 1),
            )
        else:
            # lithium concentration in particle core
            c_c_xav = pybamm.Variable(
                f"X-averaged {domain} {phase_name}core lithium concentration [mol.m-3]",
                f"{domain} {phase_name}core",
                auxiliary_domains={"secondary": "current collector"},
                bounds=(0, self.phase_param.c_max),
                scale=self.phase_param.c_max,
            )
            c_c_xav.print_name = f"c_c_{domain[0]}_xav"
            c_c = pybamm.SecondaryBroadcast(c_c_xav, f"{domain} electrode")

            # oxygen concentration in degraded shell
            c_o_xav = pybamm.Variable(
                f"X-averaged {domain} {phase_name}shell oxygen concentration [mol.m-3]",
                f"{domain} {phase_name}shell",
                auxiliary_domains={"secondary": "current collector"},
            )
            c_o = pybamm.SecondaryBroadcast(c_o_xav, f"{domain} electrode")

            # location of core-shell boundary
            s_nd_xav = pybamm.Variable(
                f"X-averaged {domain} {phase_name}particle "
                "moving phase boundary location",
                "current collector",
                bounds=(0, 1),
            )
            s_nd = pybamm.PrimaryBroadcast(s_nd_xav, f"{domain} electrode")

        # Standard concentration variables (size-independent)
        variables.update(self._get_standard_concentration_variables(c_c, c_o, s_nd))

        return variables

    def get_coupled_variables(self, variables):
        domain, Domain = self.domain_Domain
        phase_name = self.phase_name
        phase_param = self.phase_param
        param = self.param

        R_typ = phase_param.R_typ
        c_thrd = phase_param.c_c_thrd
        c_trap = phase_param.c_s_trap
        c_o_core = phase_param.c_o_core
        k_1 = phase_param.k_1
        k_2 = phase_param.k_2

        if self.x_average is False:
            # independent variables
            c_c = variables[
                f"{Domain} {phase_name}core lithium concentration [mol.m-3]"
            ]
            c_o = variables[
                f"{Domain} {phase_name}shell oxygen concentration [mol.m-3]"
            ]
            s_nd = variables[
                f"{Domain} {phase_name}particle moving phase boundary location"
            ]

            # temperature
            T = variables[f"{Domain} electrode temperature [K]"]
            T_c = pybamm.PrimaryBroadcast(T, [f"{domain} {phase_name}core"])
            T_o = pybamm.PrimaryBroadcast(T, [f"{domain} {phase_name}shell"])

            R_nd = variables[f"{Domain} {phase_name}particle radius"]
            j = variables[
                f"{Domain} electrode {phase_name}" "interfacial current density [A.m-2]"
            ]

            # variable values at the moving core-shell phase boundary
            # this is extrapolated from nodal values at cell centre
            c_o_inner = variables[
                f"{Domain} {phase_name}shell inner boundary "
                "oxygen concentration [mol.m-3]"
            ]
            c_c_outer = variables[
                f"{Domain} {phase_name}core outer boundary "
                "lithium concentration [mol.m-3]"
            ]

            c_c_N = variables[
                f"{Domain} {phase_name}core surface cell lithium concentration [mol.m-3]"
            ]
            c_o_1 = variables[
                f"{Domain} {phase_name}shell inner cell oxygen concentration [mol.m-3]"
            ]
            dx_core = variables[f"{Domain} {phase_name}core surface cell length [m]"]
            dx_shell = variables[f"{Domain} {phase_name}shell inner cell length [m]"]
        else:
            # independent variables
            c_c = variables[
                f"X-averaged {domain} {phase_name}core lithium concentration [mol.m-3]"
            ]
            c_o = variables[
                f"X-averaged {domain} {phase_name}shell oxygen concentration [mol.m-3]"
            ]
            s_nd = variables[
                f"X-averaged {domain} {phase_name}particle "
                "moving phase boundary location"
            ]

            # temperature
            T = variables[f"X-averaged {domain} electrode temperature [K]"]
            T_c = pybamm.PrimaryBroadcast(T, [f"{domain} {phase_name}core"])
            T_o = pybamm.PrimaryBroadcast(T, [f"{domain} {phase_name}shell"])

            # we use averaged particle radius to represent all particles here
            # R_nd = 1 assumes the typical radius (R value at electrode middle point)
            # represent all particles, as adopted in particle ficikan model SPM
            R_nd = variables[f"X-averaged {domain} {phase_name}particle radius"]

            j = variables[
                f"X-averaged {domain} electrode {phase_name}"
                "interfacial current density [A.m-2]"
            ]

            # variable values at the moving core-shell phase boundary
            # this is extrapolated from nodal values at cell centre
            c_o_inner = variables[
                f"X-averaged {domain} {phase_name}shell inner boundary "
                "oxygen concentration [mol.m-3]"
            ]
            c_c_outer = variables[
                f"X-averaged {domain} {phase_name}core outer boundary "
                "lithium concentration [mol.m-3]"
            ]

            c_c_N = variables[
                f"X-averaged {domain} {phase_name}core surface cell "
                "lithium concentration [mol.m-3]"
            ]
            c_o_1 = variables[
                f"X-averaged {domain} {phase_name}shell inner cell "
                "oxygen concentration [mol.m-3]"
            ]
            dx_core = variables[
                f"X-averaged {domain} {phase_name}core surface cell length [m]"
            ]
            dx_shell = variables[
                f"X-averaged {domain} {phase_name}shell inner cell length [m]"
            ]

        # time rate of moving phase boundary
        # EqualHeaviside evaluates whether left <= right
        s_nd_dot = (
            -(k_1 - k_2 * c_o_inner)
            / (R_nd * R_typ)
            * pybamm.EqualHeaviside(c_c_outer, c_thrd)
        )

        # defined in base_phase_transition
        D_c = pybamm.surf(phase_param.D(c_c, T_c))
        D_o = pybamm.boundary_value(phase_param.D_o(c_o, T_o), "left")

        # Derived variable values at the moving phase boundary
        # from applied boundary conditions of mass conservation
        # The boundary conditions for the lithium and oxygen diffusion equations
        # are of Robin type, and thus the boundary values are first calculated for
        # later enforcement in the manner of Neumann-like bc
        # Note they are not extrapolated from cell central variable values
        # lithium concentration at the moving phase boundary
        c_c_b = (
            c_c_N
            - dx_core
            / D_c
            * (R_nd / s_nd * j / param.F - (R_nd**2) * R_typ * s_nd * s_nd_dot * c_trap)
        ) / (1 + dx_core / D_c * (R_nd**2) * R_typ * s_nd * s_nd_dot)

        # oxygen concentration at the moving phase boundary
        c_o_b = (
            c_o_1
            - dx_shell / D_o * (R_nd**2) * R_typ * (1 - s_nd) * s_nd_dot * c_o_core
        ) / (1 - dx_shell / D_o * (R_nd**2) * R_typ * (1 - s_nd) * s_nd_dot)

        # boundary conditions for lithium diffusion in the core
        rbc_c_c = (c_c_b - c_c_N) / dx_core
        # boundary conditions for oxygen diffusion in the shell
        lbc_c_o = (c_o_1 - c_o_b) / dx_shell

        if self.x_average is True:
            s_nd_dot = pybamm.PrimaryBroadcast(s_nd_dot, [f"{domain} electrode"])
            c_c_b = pybamm.PrimaryBroadcast(c_c_b, [f"{domain} electrode"])
            c_o_b = pybamm.PrimaryBroadcast(c_o_b, [f"{domain} electrode"])

        variables.update(
            {
                f"Time derivative of {domain} {phase_name}particle "
                "moving phase boundary location [s-1]": s_nd_dot,
                f"X-averaged time derivative of {domain} {phase_name}particle "
                "moving phase boundary location [s-1]": pybamm.x_average(s_nd_dot),
                f"{Domain} {phase_name}particle core-shell boundary "
                "lithium concentration [mol.m-3]": c_c_b,
                f"X-averaged {domain} {phase_name}particle core-shell boundary "
                "lithium concentration [mol.m-3]": pybamm.x_average(c_c_b),
                f"{Domain} {phase_name}particle core-shell boundary "
                "oxygen concentration [mol.m-3]": c_o_b,
                f"X-averaged {domain} {phase_name}particle core-shell boundary "
                "oxygen concentration [mol.m-3]": pybamm.x_average(c_o_b),
                f"{Domain} {phase_name}core right-hand-side bc [mol.m-4]": rbc_c_c,
                f"{Domain} {phase_name}shell left-hand-side bc [mol.m-4]": lbc_c_o,
            }
        )

        return variables

    def set_rhs(self, variables):
        domain, Domain = self.domain_Domain
        phase_name = self.phase_name
        phase_param = self.phase_param
        phase = self.phase
        param = self.param

        R_typ = phase_param.R_typ

        if domain == "positive":
            options_phase = getattr(self.options, domain)["particle phases"]
        else:
            raise DomainError(
                "Domain must be 'positive' for phase transition degradation."
                "Spatial variables only defined for positive core and shell."
            )

        if self.x_average is False:
            c_c = variables[
                f"{Domain} {phase_name}core lithium concentration [mol.m-3]"
            ]
            c_o = variables[
                f"{Domain} {phase_name}shell oxygen concentration [mol.m-3]"
            ]
            s_nd = variables[
                f"{Domain} {phase_name}particle moving phase boundary location"
            ]
            # temperature
            T = variables[f"{Domain} electrode temperature [K]"]
            T_c = pybamm.PrimaryBroadcast(T, [f"{domain} {phase_name}core"])
            T_o = pybamm.PrimaryBroadcast(T, [f"{domain} {phase_name}shell"])
            R_nd = variables[f"{Domain} {phase_name}particle radius"]
            s_nd_dot = variables[
                f"Time derivative of {domain} {phase_name}particle "
                "moving phase boundary location [s-1]"
            ]
            if options_phase == "1" and phase == "primary":
                r_co = pybamm.standard_spatial_vars.r_co
                r_sh = pybamm.standard_spatial_vars.r_sh
            elif options_phase == "2" and phase == "primary":
                r_co = pybamm.standard_spatial_vars.r_co_prim
                r_sh = pybamm.standard_spatial_vars.r_sh_prim
            elif options_phase == "2" and phase == "secondary":
                r_co = pybamm.standard_spatial_vars.r_co_sec
                r_sh = pybamm.standard_spatial_vars.r_sh_sec
        else:
            c_c = variables[
                f"X-averaged {domain} {phase_name}core lithium concentration [mol.m-3]"
            ]
            c_o = variables[
                f"X-averaged {domain} {phase_name}shell oxygen concentration [mol.m-3]"
            ]
            s_nd = variables[
                f"X-averaged {domain} {phase_name}particle "
                "moving phase boundary location"
            ]
            # temperature
            T = variables[f"X-averaged {domain} electrode temperature [K]"]
            T_c = pybamm.PrimaryBroadcast(T, [f"{domain} {phase_name}core"])
            T_o = pybamm.PrimaryBroadcast(T, [f"{domain} {phase_name}shell"])
            # R_nd = variables[f"X-averaged {domain} {phase_name}particle radius"]
            R_nd = 1
            s_nd_dot = variables[
                f"X-averaged time derivative of {domain} {phase_name}particle "
                "moving phase boundary location [s-1]"
            ]
            if options_phase == "1" and phase == "primary":
                r_co = pybamm.x_average(pybamm.standard_spatial_vars.r_co)
                r_sh = pybamm.x_average(pybamm.standard_spatial_vars.r_sh)
            elif options_phase == "2" and phase == "primary":
                r_co = pybamm.x_average(pybamm.standard_spatial_vars.r_co_prim)
                r_sh = pybamm.x_average(pybamm.standard_spatial_vars.r_sh_prim)
            elif options_phase == "2" and phase == "secondary":
                r_co = pybamm.x_average(pybamm.standard_spatial_vars.r_co_sec)
                r_sh = pybamm.x_average(pybamm.standard_spatial_vars.r_sh_sec)

        D_c = phase_param.D(c_c, T_c)
        D_o = phase_param.D_o(c_o, T_o)

        self.rhs[c_c] = pybamm.inner(r_co * s_nd_dot / s_nd, pybamm.grad(c_c)) + 1 / (
            R_nd**2
        ) / (s_nd**2) * pybamm.div(D_c * pybamm.grad(c_c))
        # div((r_sh * (1 - s_nd) + s_nd * R_typ)**2 * D_o * pybamm.grad(c_o))
        # will lead to rounding errors
        # (D_o * pybamm.grad(c_o)) must be bracketed to avoid the rounding errors
        # or first define N_o to be called later, as did below
        N_o = -D_o * pybamm.grad(c_o)
        self.rhs[c_o] = pybamm.inner(
            (R_typ - r_sh) / (1 - s_nd) * s_nd_dot, pybamm.grad(c_o)
        ) - (
            1
            / ((R_nd * (1 - s_nd) * (r_sh * (1 - s_nd) + s_nd * R_typ)) ** 2)
            * pybamm.div((r_sh * (1 - s_nd) + s_nd * R_typ) ** 2 * N_o)
        )
        self.rhs[s_nd] = s_nd_dot

    def set_boundary_conditions(self, variables):
        domain, Domain = self.domain_Domain
        phase_name = self.phase_name

        if self.x_average is False:
            c_c = variables[
                f"{Domain} {phase_name}core lithium concentration [mol.m-3]"
            ]
            c_o = variables[
                f"{Domain} {phase_name}shell oxygen concentration [mol.m-3]"
            ]
        else:
            c_c = variables[
                f"X-averaged {domain} {phase_name}core lithium concentration [mol.m-3]"
            ]
            c_o = variables[
                f"X-averaged {domain} {phase_name}shell oxygen concentration [mol.m-3]"
            ]

        rbc_c_c = variables[f"{Domain} {phase_name}core right-hand-side bc [mol.m-4]"]
        lbc_c_o = variables[f"{Domain} {phase_name}shell left-hand-side bc [mol.m-4]"]

        self.boundary_conditions[c_c] = {
            "left": (pybamm.Scalar(0), "Neumann"),
            "right": (rbc_c_c, "Neumann"),
        }
        self.boundary_conditions[c_o] = {
            "left": (lbc_c_o, "Neumann"),
            "right": (pybamm.Scalar(0), "Dirichlet"),
        }

    def set_initial_conditions(self, variables):
        domain, Domain = self.domain_Domain
        phase_name = self.phase_name

        c_c_init = self.phase_param.c_c_init
        c_o_init = self.phase_param.c_o_init
        s_nd_init = self.phase_param.s_nd_init

        if self.x_average is False:
            c_c = variables[
                f"{Domain} {phase_name}core lithium concentration [mol.m-3]"
            ]
            c_o = variables[
                f"{Domain} {phase_name}shell oxygen concentration [mol.m-3]"
            ]
            s_nd = variables[
                f"{Domain} {phase_name}particle moving phase boundary location"
            ]
        else:
            c_c = variables[
                f"X-averaged {domain} {phase_name}core lithium concentration [mol.m-3]"
            ]
            c_o = variables[
                f"X-averaged {domain} {phase_name}shell oxygen concentration [mol.m-3]"
            ]
            s_nd = variables[
                f"X-averaged {domain} {phase_name}particle "
                "moving phase boundary location"
            ]
            c_c_init = pybamm.x_average(c_c_init)
            c_o_init = pybamm.x_average(c_o_init)
            s_nd_init = pybamm.x_average(s_nd_init)

        self.initial_conditions = {
            c_c: c_c_init,
            c_o: c_o_init,
            s_nd: s_nd_init,
        }
