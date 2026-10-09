#
# Base class for positive electrode degradation
#

import numpy as np

import pybamm

# Spatial variables for SPM single particle model
eta_xav = pybamm.SpatialVariable(
    "eta",
    domains={
        "primary": "positive core",
        "secondary": "current collector",
    },
    coord_sys="spherical polar",
)

psi_xav = pybamm.SpatialVariable(
    "psi",
    domains={
        "primary": "positive shell oxygen",
        "secondary": "current collector",
    },
    coord_sys="cartesian",
)


class BasePositiveElectrodeDegradation(pybamm.BaseSubModel):
    """
    Base class for the shrinking-core degradation of the positive electrode.
    When oxygen is being released the core shrinks as a shell grows from its surface inwards
    inward :footcite:t:`Ghosh2021`, causing loss of active material and of
    cyclable lithium :footcite:t:`Zhuo2023`.

    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel
    domain : str
        The domain of the model, which must be "Positive"

    **Extends:** :class:`pybamm.BaseSubModel`
    """

    def __init__(self, param, domain):
        super().__init__(param, domain)
        pybamm.citations.register("Ghosh2021")
        pybamm.citations.register("Zhuo2023")

        c_o_core_dim = pybamm.Parameter(
            "Constant oxygen concentration in particle core [mol.m-3]"
        )
        c_p_thrd_dim = pybamm.Parameter(
            "Threshold concentration for phase transition [mol.m-3]"
        )
        c_s_trap_dim = pybamm.Parameter(
            "Trapped lithium concentration in shell [mol.m-3]"
        )
        c_c_bott_dim = pybamm.Parameter(
            "Minimum concentration in positive core when fully charged [mol.m-3]"
        )
        c_n_bott_dim = pybamm.Parameter(
            "Minimum concentration in negative particle when fully discharged [mol.m-3]"
        )

        self.c_o_core_dim = c_o_core_dim  # self.param.p.prim.c_max
        self.c_o_core = c_o_core_dim / c_o_core_dim

        self.c_p_thrd = c_p_thrd_dim / self.param.p.prim.c_max
        self.c_s_trap = c_s_trap_dim / self.param.p.prim.c_max
        self.c_c_bott = c_c_bott_dim / self.param.p.prim.c_max
        self.c_n_bott = c_n_bott_dim / self.param.n.prim.c_max

    def _get_standard_concentration_variables(
        self, c_c, c_o, s, c_c_xav=None, c_o_xav=None, s_xav=None
    ):
        """
        All positive electrode degradation submodels must provide the core concentration,
        oxygen concentration in shell, and phase boundary location
        as arguments.
        """

        # Get surface and center concentration if not provided as fundamental
        # variable to solve for
        c_c_surf = pybamm.surf(c_c)
        c_c_surf_av = pybamm.x_average(c_c_surf)

        c_o_surf = pybamm.surf(c_o)
        c_o_surf_av = pybamm.x_average(c_o_surf)
        c_o_cent = pybamm.boundary_value(c_o, "left")
        c_o_cent_av = pybamm.x_average(c_o_cent)

        # Maximum concentration as the reference
        c_scale = self.param.p.prim.c_max

        # Particle radius at each through-cell position [m].
        R_x = self.R_p_dimensional(pybamm.standard_spatial_vars.x_p)
        eta_reciprocal = s / R_x

        # Get average concentration
        c_c_xav = pybamm.x_average(c_c) if c_c_xav is None else c_c_xav
        c_c_rav = self._pe_r_average(c_c, eta_reciprocal)
        c_c_av = pybamm.x_average(c_c_rav)

        c_o_xav = pybamm.x_average(c_o) if c_o_xav is None else c_o_xav
        c_o_rav = self._pe_r_average(c_o, eta_reciprocal)

        s_xav = pybamm.x_average(s) if s_xav is None else s_xav

        # Boundary cell value
        # Surface (rightmost) cell for core c_c
        c_c_N = pybamm.boundary_cell_value(c_c, "right")
        c_c_N_av = pybamm.x_average(c_c_N)

        # center (leftmost) cell for shell c_o
        c_o_1 = pybamm.boundary_cell_value(c_o, "left")
        c_o_1_av = pybamm.x_average(c_o_1)

        # boundary cell length: node to the edge
        dx_cp = pybamm.boundary_cell_length(c_c, "right")
        dx_cp_av = pybamm.x_average(dx_cp)

        dx_co = pybamm.boundary_cell_length(c_o, "left")
        dx_co_av = pybamm.x_average(dx_co)

        # Loss of active material in Positive Electrode
        lam_pe = pybamm.Scalar(1) - (eta_reciprocal) ** 3
        lam_pe_av = pybamm.x_average(lam_pe)

        variables = {
            # Core concentration
            "Positive core stoichiometry": c_c,
            "Positive core concentration [mol.m-3]": c_c * c_scale,
            "X-averaged positive core stoichiometry": c_c_xav,
            "X-averaged positive core concentration [mol.m-3]": c_c_xav * c_scale,
            "R-averaged positive core stoichiometry": c_c_rav,
            "R-averaged positive core concentration [mol.m-3]": c_c_rav * c_scale,
            "Average positive core stoichiometry": c_c_av,
            "Average positive core concentration [mol.m-3]": c_c_av * c_scale,
            "Positive core surface stoichiometry": c_c_surf,
            "Positive core surface concentration [mol.m-3]": c_scale * c_c_surf,
            "X-averaged positive core surface stoichiometry": c_c_surf_av,
            "X-averaged positive core surface concentration [mol.m-3]": c_scale
            * c_c_surf_av,
            "Minimum positive core stoichiometry": pybamm.min(c_c),
            "Maximum positive core stoichiometry": pybamm.max(c_c),
            "Minimum positive core concentration [mol.m-3]": pybamm.min(c_c) * c_scale,
            "Maximum positive core concentration [mol.m-3]": pybamm.max(c_c) * c_scale,
            "Minimum positive core surface stoichiometry": pybamm.min(c_c_surf),
            "Maximum positive core surface stoichiometry": pybamm.max(c_c_surf),
            "Minimum positive core surface concentration [mol.m-3]": pybamm.min(
                c_c_surf
            )
            * c_scale,
            "Maximum positive core surface concentration [mol.m-3]": pybamm.max(
                c_c_surf
            )
            * c_scale,
            # Shell concentration of oxygen
            "Positive shell oxygen stoichiometry": c_o,
            "Positive shell concentration of oxygen [mol.m-3]": c_o * self.c_o_core_dim,
            "X-averaged positive shell oxygen stoichiometry": c_o_xav,
            "X-averaged positive shell concentration of oxygen [mol.m-3]": c_o_xav
            * self.c_o_core_dim,
            "R-averaged positive shell oxygen stoichiometry": c_o_rav,
            "R-averaged positive shell concentration of oxygen [mol.m-3]": c_o_rav
            * self.c_o_core_dim,
            "Positive shell surface oxygen stoichiometry": c_o_surf,
            "X-averaged positive shell surface oxygen stoichiometry": c_o_surf_av,
            "Positive shell center oxygen stoichiometry": c_o_cent,
            "Positive shell center concentration of oxygen [mol.m-3]": c_o_cent
            * self.c_o_core_dim,
            "X-averaged positive shell center oxygen stoichiometry": c_o_cent_av,
            "X-averaged positive shell center concentration of oxygen [mol.m-3]": c_o_cent_av
            * self.c_o_core_dim,
            # Moving phase boundary
            "Moving phase boundary location [m]": s,
            "Moving phase boundary location": eta_reciprocal,
            "X-averaged moving phase boundary location [m]": s_xav,
            "X-averaged moving phase boundary location": pybamm.x_average(
                eta_reciprocal
            ),
            # loss of active material (LAM) due to progressing of s
            # The shell is considered as LAM
            "Positive particle shell volume fraction": lam_pe,
            "X-averaged positive particle shell volume fraction": lam_pe_av,
            # Boundary cell value and length (geometry)
            "Positive core surface cell stoichiometry": c_c_N,
            "Positive core surface cell concentration [mol.m-3]": c_scale * c_c_N,
            "Positive shell center cell oxygen stoichiometry": c_o_1,
            "Positive shell center cell concentration of oxygen [mol.m-3]": c_o_1
            * self.c_o_core_dim,
            "X-averaged positive core surface cell stoichiometry": c_c_N_av,
            "X-averaged positive core surface cell concentration [mol.m-3]": c_scale
            * c_c_N_av,
            "X-averaged positive shell center cell oxygen stoichiometry": c_o_1_av,
            "X-averaged positive shell center cell concentration of oxygen [mol.m-3]": c_o_1_av
            * self.c_o_core_dim,
            # Half cell length: center node to edge
            "Positive core surface cell length": dx_cp,
            "Positive shell center cell length of oxygen": dx_co,
            "X-averaged positive core surface cell length": dx_cp_av,
            "X-averaged positive shell center cell length of oxygen": dx_co_av,
            # Particle surface
            "Positive particle surface concentration [mol.m-3]": c_scale * c_c_surf,
            "R-averaged positive particle concentration [mol.m-3]": (
                c_c_rav * (eta_reciprocal) ** 3
                + self.c_s_trap * (1 - (eta_reciprocal) ** 3)
            )
            * c_scale,
            "Positive particle surface stoichiometry": c_c_surf,
        }

        return variables

    def _pe_r_average(self, symbol, s):
        """
        Volume average over the transformed radial coordinate.
        """
        if symbol.domain == ["positive core"]:
            eta = pybamm.SpatialVariable("eta", domains=symbol.domains)
            v = pybamm.FullBroadcast(pybamm.Scalar(1), broadcast_domains=symbol.domains)
            return pybamm.Integral(symbol, eta) / pybamm.Integral(v, eta)

        elif symbol.domain in [["positive shell"], ["positive shell oxygen"]]:
            chi = pybamm.SpatialVariable("chi", domains=symbol.domains)
            v = pybamm.FullBroadcast(pybamm.Scalar(1), broadcast_domains=symbol.domains)
            coeff = 4 * np.pi * ((1 - s) * chi + s) ** 2 * (1 - s)
            return pybamm.Integral(coeff * symbol, chi) / pybamm.Integral(
                coeff * v, chi
            )
        else:
            raise pybamm.DomainError("domain must be positive core or shell (oxygen).")

    def _get_total_concentration_variables(self, variables):
        """
        Sum the total and cyclable lithium in each electrode to `variables.
        Note the lithium trapped in the positive electrode shell is not cyclable.
        """
        s = variables["Moving phase boundary location"]
        c_c_rav = variables["R-averaged positive core stoichiometry"]

        eps_p = variables["Positive electrode active material volume fraction"]
        eps_p_av = pybamm.x_average(eps_p)

        lam_pe_av = variables["X-averaged positive particle shell volume fraction"]

        # Total lithium in the particle = core + shell.
        c_c_vol_av = (
            pybamm.x_average(eps_p * (c_c_rav * s**3 + self.c_s_trap * (1 - s**3)))
            / eps_p_av
        )

        # Total cyclable lithium in core
        c_c_vol_av_cyc = pybamm.x_average(eps_p * (c_c_rav - self.c_c_bott)) / eps_p_av
        c_scale = self.param.p.prim.c_max

        # Total cyclable lithium in negative particle
        c_n_rav = variables["R-averaged negative particle concentration"]
        eps_n = variables["Negative electrode active material volume fraction"]
        eps_n_av = pybamm.x_average(eps_n)
        c_n_vol_av_cyc = pybamm.x_average(eps_n * (c_n_rav - self.c_n_bott)) / eps_n_av

        # Positive electrode thickness [m]
        L = self.param.p.L
        # Area of current collector
        A = self.param.A_cc

        variables.update(
            {
                "Positive electrode volume-averaged stoichiometry": c_c_vol_av,
                "Positive electrode volume-averaged concentration [mol.m-3]": c_c_vol_av
                * c_scale,
                "Total lithium in positive electrode [mol]": pybamm.yz_average(
                    c_c_vol_av
                    * c_scale
                    * L
                    * A
                    * eps_p_av  # Positive electrode degradation active material volume
                ),
                "Total cyclable lithium in positive electrode [mol]": pybamm.yz_average(
                    c_c_vol_av_cyc
                    * c_scale
                    * L
                    * A
                    * eps_p_av  # Positive electrode  active material volume
                    * (1 - lam_pe_av)
                ),
                "Total cyclable lithium in negative electrode [mol]": pybamm.yz_average(
                    c_n_vol_av_cyc
                    * self.param.n.prim.c_max
                    * self.param.n.L
                    * A
                    * eps_n_av  # Negative electrode active material volume
                ),
            }
        )
        return variables

    # Define parameters that are exclusive to the positive electrode degradation model
    # Positive core will use already defined positive particle

    def D_c_dimensional(self, sto, T):
        """Dimensional diffusivity in positive core"""
        inputs = {"Positive core stoichiometry": sto, "Temperature [K]": T}
        return pybamm.FunctionParameter("Positive core diffusivity [m2.s-1]", inputs)

    def D_o_dimensional(self, sto, T):
        """Dimensional oxygen diffusivity in positive shell"""
        inputs = {"Positive shell oxygen stoichiometry": sto, "Temperature [K]": T}
        return pybamm.FunctionParameter(
            "Positive shell oxygen diffusivity [m2.s-1]", inputs
        )

    def k_1_dimensional(self, T):
        """Dimensional forward chemical reaction coefficient"""
        inputs = {"Temperature [K]": T}
        return pybamm.FunctionParameter(
            "Forward chemical reaction coefficient [m.s-1]", inputs
        )

    def k_2_dimensional(self, T):
        """Dimensional reverse chemical reaction coefficient"""
        inputs = {"Temperature [K]": T}
        return pybamm.FunctionParameter(
            "Reverse chemical reaction coefficient [m4.mol-1.s-1]", inputs
        )

    def c_c_init_dimensional(self, x):
        """Initial concentration as a function of dimensionless position x"""
        inputs = {"Dimensionless through-cell position (x_p)": x}
        return pybamm.FunctionParameter(
            "Initial concentration in positive core [mol.m-3]", inputs
        )

    def c_c_init(self, x):
        """
        Dimensionless initial concentration as a function of dimensionless position x
        """
        return self.c_c_init_dimensional(x) / self.param.p.prim.c_max

    def c_o_init_dimensional(self, psi):
        """Initial oxygen concentration as a function of dimensionless position x"""
        inputs = {"Dimensionless transformed position shell oxygen (psi)": psi}
        return pybamm.FunctionParameter(
            "Initial oxygen concentration in positive shell [mol.m-3]", inputs
        )

    def c_o_init(self, psi):
        """
        Dimensionless initial oxygen concentration as a function of dimensionless position x
        """
        return self.c_o_init_dimensional(psi) / self.c_o_core_dim

    def s_init_dimensional(self, x):
        """
        Initial phase boundary location at through-cell position x [m].
        """
        inputs = {"Through-cell distance (x) [m]": x}
        return pybamm.FunctionParameter("Initial phase boundary location [m]", inputs)

    def R_p_dimensional(self, x):
        """
        Positive particle radius at through-cell position x [m].
        """
        inputs = {"Through-cell distance (x) [m]": x}
        return pybamm.FunctionParameter("Positive particle radius [m]", inputs)

    def s_init(self, x):
        """
        Phase boundary location s at through-cell position x [m]
        """
        return self.s_init_dimensional(x)
