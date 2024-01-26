#
# Base class for the core-shell model for Positive Electrode (NMC811)
# phase transition cased degradation, based on paper
#    Mingzhao Zhuo, Gregory Offer, Monica Marinescu, "Degradation model of
#    high-nickel positive electrodes: Effects of loss of active material and
#    cyclable lithium on capacity fade", Journal of Power Sources,
#    556 (2023): 232461. doi: 10.1016/j.jpowsour.2022.232461.
#
import numpy as np
import pybamm

class BasePhaseTransition(pybamm.BaseSubModel):
    """
    Base class for the core-shell model.

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

    References
    ----------
    .. [1] Mingzhao Zhuo, Gregory Offer, Monica Marinescu, "Degradation model of
           high-nickel positive electrodes: Effects of loss of active material 
           and cyclable lithium on capacity fade", Journal of Power Sources}, 
           556 (2023): 232461.
           doi: 10.1016/j.jpowsour.2022.232461.
    """

    def __init__(self, param, domain, options, phase="primary"):
        super().__init__(param, domain, options=options, phase=phase)

        # Read from options to see if we have a particle size distribution
        domain_options = getattr(self.options, domain)
        self.size_distribution = domain_options["particle size"] == "distribution"
        if self.size_distribution is True:
            raise NotImplementedError(
                "PE phase transition submodel only implemented for particles "
                "of single size, i.e, no distribution."
            )

    def _get_standard_concentration_variables(
        self, c_c, c_o, s_nd,
        c_c_xav=None, c_o_xav=None, s_nd_av=None
    ):
        """
        Provide the lithium concentration in the core,
        oxygen concentration in the shell, and
        core-shell phase boundary location
        as arguments.
        """
        domain, Domain = self.domain_Domain
        phase_name = self.phase_name

        # Get outer (surface) and inner boundary concentration if not provided 
        # as fundamental variable to solve for
        c_c_surf     = pybamm.surf(c_c)
        c_c_surf_av  = pybamm.x_average(c_c_surf)
        c_c_outer    = c_c_surf
        c_c_outer_av = c_c_surf_av

        c_o_surf     = pybamm.surf(c_o)
        c_o_surf_av  = pybamm.x_average(c_o_surf)
        c_o_outer    = c_o_surf
        c_o_outer_av = c_o_surf_av
        c_o_inner    = pybamm.boundary_value(c_o, "left")
        c_o_inner_av = pybamm.x_average(c_o_inner)

        # Get average concentration(s) if not provided as fundamental variable to
        # solve for
        if c_c_xav is None:
            c_c_xav = pybamm.x_average(c_c)
        # c_c_rav = self._r_average_core(c_c)
        # the r_average can also be calculated from normal RAverage
        c_c_rav = pybamm.r_average(c_c)
        c_c_av  = pybamm.x_average(c_c_rav)

        if c_o_xav is None:
            c_o_xav = pybamm.x_average(c_o)
        c_o_rav = self._r_average_shell(c_o, s_nd)
        c_o_av  = pybamm.x_average(c_o_rav)

        if s_nd_av is None:
            s_nd_av = pybamm.x_average(s_nd)

        # concentration nodal value at outermost cell
        c_c_N    = pybamm.boundary_cell_value(c_c, "right")
        c_c_N_av = pybamm.x_average(c_c_N)

        # concentration nodal value at innermost cell
        c_o_1    = pybamm.boundary_cell_value(c_o, "left")
        c_o_1_av = pybamm.x_average(c_o_1)

        # boundary cell length: node to the edge (half cell length)
        dx_core    = pybamm.boundary_cell_length(c_c, "right")
        dx_core_av = pybamm.x_average(dx_core)

        dx_shell    = pybamm.boundary_cell_length(c_o, "left")
        dx_shell_av = pybamm.x_average(dx_shell)

        # Definition of loss of active material in PE
        # that is, the fraction of shell phase
        lam_pe    = pybamm.Scalar(1) - s_nd ** 3
        lam_pe_av = pybamm.x_average(lam_pe)

        c_scale = self.phase_param.c_max

        variables = {
            # Dimensional lithium concentration in the core
            f"{Domain} {phase_name}core lithium concentration [mol.m-3]": c_c,
            f"X-averaged {domain} {phase_name}core "
            "lithium concentration [mol.m-3]": c_c_xav,
            f"R-averaged {domain} {phase_name}core "
            "lithium concentration [mol.m-3]": c_c_rav,
            f"Average {domain} {phase_name}core "
            "lithium concentration [mol.m-3]": c_c_av,
            f"{Domain} {phase_name}core surface "
            "lithium concentration [mol.m-3]": c_c_surf,
            f"X-averaged {domain} {phase_name}core surface"
            "lithium concentration [mol.m-3]": c_c_surf_av,
            f"{Domain} {phase_name}core outer boundary "
            "lithium concentration [mol.m-3]": c_c_outer,
            f"X-averaged {domain} {phase_name}core outer boundary "
            "lithium concentration [mol.m-3]": c_c_outer_av,
            f"Minimum {domain} {phase_name}core concentration [mol.m-3]"
            "": pybamm.min(c_c),
            f"Maximum {domain} {phase_name}core concentration [mol.m-3]"
            "": pybamm.max(c_c),
            f"Minimum {domain} {phase_name}core "
            "surface concentration [mol.m-3]": pybamm.min(c_c_surf),
            f"Maximum {domain} {phase_name}core "
            "surface concentration [mol.m-3]": pybamm.max(c_c_surf),
            # Dimensionless concentration
            f"{Domain} {phase_name}core lithium concentration": c_c / c_scale,
            f"X-averaged {domain} {phase_name}core lithium concentration": c_c_xav
            / c_scale,
            f"R-averaged {domain} {phase_name}core lithium concentration": c_c_rav
            / c_scale,
            f"Average {domain} {phase_name}core lithium concentration": c_c_av / c_scale,
            f"{Domain} {phase_name}core surface lithium concentration": c_c_surf / c_scale,
            f"X-averaged {domain} {phase_name}core "
            "surface lithium concentration": c_c_surf_av / c_scale,
            # Stoichiometry (equivalent to dimensionless concentration)
            f"{Domain} {phase_name}core lithium stoichiometry": c_c / c_scale,
            f"X-averaged {domain} {phase_name}core lithium stoichiometry": c_c_xav
            / c_scale,
            f"R-averaged {domain} {phase_name}core lithium stoichiometry": c_c_rav
            / c_scale,
            f"Average {domain} {phase_name}core lithium stoichiometry": c_c_av / c_scale,
            f"{Domain} {phase_name}core surface lithium stoichiometry": c_c_surf / c_scale,
            f"X-averaged {domain} {phase_name}core "
            "surface lithium stoichiometry": c_c_surf_av / c_scale,
            # Dimensional oxygen concentration in the shell
            f"{Domain} {phase_name}shell oxygen concentration [mol.m-3]"
            "": c_o,
            f"X-averaged {domain} {phase_name}shell "
            "oxygen concentration [mol.m-3]": c_o_xav,
            f"R-averaged {domain} {phase_name}shell "
            "oxygen concentration [mol.m-3]": c_o_rav,
            f"Average {domain} {phase_name}shell "
            "oxygen concentration [mol.m-3]": c_o_av,
            f"{Domain} {phase_name}shell inner boundary "
            "oxygen concentration [mol.m-3]": c_o_inner,
            f"X-averaged {domain} {phase_name}shell inner boundary "
            "oxygen concentration [mol.m-3]": c_o_inner_av,
            f"{Domain} {phase_name}shell surface "
            "oxygen concentration [mol.m-3]": c_o_surf,
            f"X-averaged {domain} {phase_name}shell surface"
            "oxygen concentration [mol.m-3]": c_o_surf_av,
            # Core-shell phase boundary
            f"{Domain} {phase_name}particle moving phase boundary location"
            "": s_nd,
            f"X-averaged {domain} {phase_name}particle "
            "moving phase boundary location": s_nd_av,
            # Boundary cell values
            f"{Domain} {phase_name}core surface cell "
            "lithium concentration [mol.m-3]": c_c_N,
            f"X-averaged {domain} {phase_name}core surface cell "
            "lithium concentration [mol.m-3]": c_c_N_av,
            f"{Domain} {phase_name}shell inner cell "
            "oxygen concentration [mol.m-3]": c_o_1,
            f"X-averaged {domain} {phase_name}shell inner cell "
            "oxygen concentration [mol.m-3]": c_o_1_av,
            # Boundary half cell length (central node to edge)
            f"{Domain} {phase_name}core surface cell length [m]": dx_core,
            f"X-averaged {domain} {phase_name}core surface cell length [m]"
            "": dx_core_av,
            f"{Domain} {phase_name}shell inner cell length [m]": dx_shell,
            f"X-averaged {domain} {phase_name}shell inner cell length [m]"
            "": dx_shell_av,
            # Loss of active material (LAM) due to progressing of s
            # the shell is considered as LAM
            "Loss of active material due to PE phase transition": lam_pe,
            "X-averaged loss of active material due to PE phase transition"
            "": lam_pe_av,
            # ---------------------------------------------------------------------
            # We need to supply a variable to
            # submodel 'positive interface' (set_interfacial_submodel) 
            # for voltage calculation
            f"{Domain} {phase_name}particle surface concentration [mol.m-3]"
            "": c_c_surf,
            # For submodel 'positive primary open-circuit potential'
            # f"{Domain} {phase_name}core surface lithium stoichiometry"
            f"{Domain} {phase_name}particle surface stoichiometry"
            "": c_c_surf / c_scale,
            # ---------------------------------------------------------------------
        }

        return variables


    def _r_average_core(self, symbol):
        domain = self.domain

        if symbol.domain != [] and symbol.domain[0].endswith("core"):
            r_co = pybamm.SpatialVariable("r_co", symbol.domain)
            # v = pybamm.FullBroadcast(
            #     pybamm.Scalar(1), broadcast_domains = symbol.domains,
            # )
            v = pybamm.ones_like(symbol)
            # cartesian coordinate
            # coeff = 4 * np.pi * r_co**2 * (R_nd * s_nd)**3
            # return pybamm.Integral(coeff * symbol, r_co) / 
            #        pybamm.Integral(coeff * v, r_co)
            # spherical polar
            # this is no different from normal pybamm.r_average, except for 
            # a coefficient s_nd^3, which is cancelled out after divided by 
            # the core volume
            return pybamm.Integral(symbol, r_co) / pybamm.Integral(v, r_co)
        else:
            raise pybamm.DomainError(
                "Domain must include 'core' for the core-shell model."
            )

    def _r_average_shell(self, symbol, s_nd):
        domain = self.domain

        if symbol.domain != [] and symbol.domain[0].endswith("shell"):
            # need to inherit the secondary domains for the multiplication with
            # s_nd with primary domain of 'positive electrode'
            r_sh = pybamm.SpatialVariable("r_sh", domains=symbol.domains)
            v = pybamm.ones_like(symbol)
            R_typ = self.phase_param.R_typ
            # exact coefficient
            # coeff = 4 * np.pi * (R_nd ** 3) * (1 - s_nd) * (
            #     ((1 - s_nd) * r_sh + s_nd * R_typ)**2
            # )
            # simplified by removing constants for the integral
            coeff = (((1 - s_nd) * r_sh + s_nd * R_typ)**2)
            return (
                pybamm.Integral(coeff * symbol, r_sh) / 
                pybamm.Integral(coeff * v, r_sh)
            )
        else:
            raise pybamm.DomainError(
                "Domain must include 'shell' for the core-shell model."
            )
