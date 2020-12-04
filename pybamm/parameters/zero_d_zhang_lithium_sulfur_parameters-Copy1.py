#
# Standard parameters for lithium-sulfur battery models
#
import pybamm


class ZeroDZhangLithiumSulfurParameters:
    """
    Standard parameters for lithium-sulfur battery models
    """

    def __init__(self):

        # Physical constants
        self.R = pybamm.constants.R
        self.F = pybamm.constants.F
        self.N = pybamm.constants.R/pybamm.constants.k_b
        self.T_ref = pybamm.Parameter("Reference temperature [K]")

        # Model-specific known parameters
        self.Ms = pybamm.Parameter("Molar mass of S8 [g.mol-1]")
        self.ns = pybamm.Parameter("Number of S atoms in S [atoms]")
        self.ns2 = pybamm.Parameter("Number of S atoms in S2 [atoms]")
        self.ns4 = pybamm.Parameter("Number of S atoms in S4 [atoms]")
        self.ns8 = pybamm.Parameter("Number of S atoms in S8 [atoms]")
        self.ne = pybamm.Parameter("Electron number per reaction [electrons]")
        self.ih0 = pybamm.Parameter("Exchange current density H [A.m-2]")
        self.il0 = pybamm.Parameter("Exchange current density L [A.m-2]")
        self.m_s = pybamm.Parameter("Mass of active sulfur per cell [g]")
        self.rho_s = pybamm.Parameter("Density of precipitated Sulfur [g.L-1]")
        self.EH0 = pybamm.Parameter("Standard Potential H [V]")
        self.EL0 = pybamm.Parameter("Standard Potential L [V]")

        self.voltage_low_cut = pybamm.Parameter("Lower voltage cut-off [V]")
        self.voltage_high_cut = pybamm.Parameter("Upper voltage cut-off [V]")
        self.n_cells = pybamm.Parameter(
            "Number of cells connected in series to make a battery"
        )

        # Model-specific unknown parameters
        self.v = pybamm.Parameter("Electrolyte volume per cell [L]")
        self.ar = pybamm.Parameter("Active reaction area per cell [m2]")
        self.k_p = pybamm.Parameter("Precipitation rate [s-1]")
        self.S_star = pybamm.Parameter("S saturation mass [g]")
        self.k_s_charge = pybamm.Parameter(
            "Shuttle rate coefficient during charge [s-1]"
        )
        self.k_s_discharge = pybamm.Parameter(
            "Shuttle rate coefficient during discharge [s-1]"
        )
        self.f_s = pybamm.Parameter("Loss rate due to shuttle [s-1]")
        self.c_h = pybamm.Parameter("Cell heat capacity [J.g-1.K-1]")
        self.A = pybamm.Parameter("Pre-Exponential factor in Arrhenius Equation [J.mol-1]")
        self.h = pybamm.Parameter("Cell heat transfer coefficient [W.K-1]")
        self.m_c = pybamm.Parameter("Cell mass [kg]")

        # Current
        # Note: pybamm.t is non-dimensional so we need to multiply by the model
        # timescale. Since the lithium-sulfur models are written in dimensional
        # form the timescale is just 1s.
        self.timescale = 1
        self.dimensional_current_with_time = pybamm.FunctionParameter(
            "Current function [A]", {"Time[s]": pybamm.t * self.timescale}
        )

        # Initial conditions defined as parameter objects
        self.S8_initial = pybamm.Parameter("Initial Condition for S8 ion [g]")
        self.S4_initial = pybamm.Parameter("Initial Condition for S4 ion [g]")
        self.S2_initial = pybamm.Parameter("Initial Condition for S2 ion [g]")
        self.S_initial = pybamm.Parameter("Initial Condition for S ion [g]")
        self.Sp_initial = pybamm.Parameter(
            "Initial Condition for Precipitated Sulfur [g]"
        )
        self.Ss_initial = pybamm.Parameter(
            "Initial Condition for Shuttled Sulfur [g]"
        )
        self.V_initial = pybamm.Parameter("Initial Condition for Terminal Voltage [V]")

        # Standard charge initial conditions
        # self.S8_initial_charge = 7.701508424389238e-12
        # self.S4_initial_charge = 0.004128610073594744
        # self.S2_initial_charge = 1.3492829940762894
        # self.S_initial_charge = 0.00012808077128429035
        # self.Sp_initial_charge = 1.3464603150711534
        # self.V_initial_charge = 2.254181286879344
