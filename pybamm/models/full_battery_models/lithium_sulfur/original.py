#
# Marinescu et al (2016) Li-S model
#
import pybamm


class OriginalMarinescuEtAl2016(pybamm.BaseModel):
    """
    Zero Dimensional model from Marinescu et al (2016) [1]. Includes S8, S4, S2, S,
    precipitated Li2S (written Sp), and voltage V as direct outputs.

    Parameters
    ----------
    options : dict, optional
        A dictionary of options to be passed to the model.
    name : str, optional
        The name of the model.

    References
    ----------
    .. [1]  Marinescu, M., Zhang, T. & Offer, G. J. (2016).
            A zero dimensional model of lithium-sulfur batteries during charge
            and discharge. Physical Chemistry Chemical Physics, 18, 584-593.

    .. [2]  Marinescu, M., O’Neill, L., Zhang, T., Walus, S., Wilson, T. E., & Offer, G. J. (2018).
            Irreversible vs reversible capacity fade of lithium-sulfur batteries during cycling: the
            effects of precipitation and shuttle. Journal of The Electrochemical Society, 165(1),
            A6107-A6118.

    """

    def __init__(self, options=None, name="Marinescu et al. (2016) model"):
        super().__init__(name)

        V = pybamm.Variable("Terminal voltage [V]")
        self.variables = {"Terminal voltage [V]": V}

        # If no options are passed, default to current control
        self.options = options or {"operating mode": "current"}

        # Set up current operated. Here the current is just provided as a
        # parameter
        if self.options["operating mode"] == "current":
            I = pybamm.Parameter("Current [A]")
            self.variables.update({"Current [A]": I})

        # If the the operating mode of the simulation is "with experiment" the
        # model option "operating mode" is the callable function
        # 'constant_current_constant_voltage_constant_power'
        elif callable(self.options["operating mode"]):
            # For the experiment we solve an extra algebraic equation
            # to determine the current
            I = pybamm.Variable("Current [A]")
            self.variables.update({"Current [A]": I})
            control_function = options["operating mode"]

            # 'constant_current_constant_voltage_constant_power' is a function
            # of current and voltage via the variables dict (see pybamm.Simulation)
            self.algebraic = {I: control_function(self.variables)}
            self.initial_conditions[I] = pybamm.Parameter("Current [A]")
        else:
            raise pybamm.OptionError()

        self.param = None
        self.submodels = {}
        self._built = False

        #######################################
        # model parameters
        #######################################
        """
        Model parameters as defined in table (1) in [1].
        We have taken standard pybamm values where possible.
        Parameters with 'H' or 'L' in the name represent the
        high and low plateau parameter, respectively.
        """

        # dynamic parameters
        """
        These values are set as model variables rather than model parameters
        """

        # standard parameters
        R = pybamm.parameters.constants.R
        F = pybamm.parameters.constants.F
        T = pybamm.Parameter("Temperature [K]")

        # model-specific known parameters
        Ms = pybamm.Parameter("Molar mass of S8 [g.mol-1]")
        ns = pybamm.Parameter("Number of S atoms in S [atoms]")
        ns2 = pybamm.Parameter("Number of S atoms in S2 [atoms]")
        ns4 = pybamm.Parameter("Number of S atoms in S4 [atoms]")
        ns8 = pybamm.Parameter("Number of S atoms in S8 [atoms]")
        ne = pybamm.Parameter("Electron number per reaction [electrons]")
        ih0 = pybamm.Parameter("Exchange current density H [A.m-2]")
        il0 = pybamm.Parameter("Exchange current density L [A.m-2]")
        m_s = pybamm.Parameter("Mass of active sulfur per cell [g]")
        rho_s = pybamm.Parameter("Density of precipitated Sulfur [g.L-1]")
        EH0 = pybamm.Parameter("Standard Potential H [V]")
        EL0 = pybamm.Parameter("Standard Potential L [V]")
        V_min = pybamm.Parameter("Lower voltage cutoff [V]")
        V_max = pybamm.Parameter("Upper voltage cutoff [V]")

        # model-specific unknown parameters
        v = pybamm.Parameter("Electrolyte volume per cell [L]")
        ar = pybamm.Parameter("Active reaction area per cell [m2]")
        k_p = pybamm.Parameter("Precipitation rate [s-1]")
        S_star = pybamm.Parameter("S saturation mass [g]")
        k_s_charge = pybamm.Parameter("Shuttle rate coefficient during charge [s-1]")
        k_s_discharge = pybamm.Parameter(
            "Shuttle rate coefficient during discharge [s-1]"
        )

        # extra
        num_cells = pybamm.Parameter(
            "Number of cells connected in series to make a battery"
        )

        # computational parameters
        """
        These parameters are groupings used for the code specifically, therefore these
        terms will not be defined as parameters.
        """
        i_coef = ne * F / (2 * R * T)
        E_H_coef = R * T / (4 * F)
        f_h = (ns4 ** 2) * Ms * v / ns8
        f_l = (ns ** 2) * ns2 * Ms ** 2 * (v ** 2) / ns4

        #######################################################
        # Non-dynamic model functions
        #######################################################

        def E_H(data):
            """
            High plateau potenital [V] as defined by equation (2a) in [1]

            Input
            -------
            data : tuple
                Contains species S8, S4, S2, S, Sp, and voltage V. Each species term and
                the voltage can be a scalar or array.

            Output
            -------
            High plateau potential [V] : pybamm.Variable

            """
            # unpack data list
            S8, S4, S2, S, Sp, V = data

            return EH0 + E_H_coef * pybamm.log(f_h * S8 / (S4 ** 2))

        def E_L(data):
            """
            Low plateau potenital [V] as defined by equation (2b) in [1]

            Inputs
            -------
            data : tuple
                Contains species S8, S4, S2, S, Sp, and voltage V. Each species term and
                the voltage can be a scalar or array.

            Output
            -------
            Low plateau potential [V] : pybamm.Variable

            """

            # unpack data list
            S8, S4, S2, S, Sp, V = data

            return EL0 + E_H_coef * pybamm.log(f_l * S4 / (S2 * (S ** 2)))

        # Surface Overpotentials

        def eta_H(data):
            """
            High plateau over-potenital [V] as defined by equation (6a) in [1]

            Input
            -------
            data : tuple
                Contains species S8, S4, S2, S, Sp, and voltage V. Each species term and
                the voltage can be a scalar or array.

            Output
            -------
            High plateau over-potential [V] : pybamm.Variable

            """

            # unpack data list
            S8, S4, S2, S, Sp, V = data

            return V - E_H(data)

        def eta_L(data):
            """
            Low plateau over-potenital [V] as defined by equation (6b) in [1]

            Input
            -------
            data : tuple
                Contains species S8, S4, S2, S, Sp, and voltage V. Each species term and
                the voltage can be a scalar or array.

            Output
            -------
            Low plateau over-potential [V] : pybamm.Variable

            """

            # unpack data list
            S8, S4, S2, S, Sp, V = data

            return V - E_L(data)

        # Half-cell Currents

        def i_H(data):
            """
            High plateau current [A] as defined by equation (5a) in [1]

            Input
            -------
            data : tuple
                Contains species S8, S4, S2, S, Sp and voltage V. Each species term and
                the voltage can be a scalar or array.

            Output
            -------
            High plateau current [A] : pybamm.Variable

            """

            S8, S4, S2, S, Sp, V = data

            return -2 * ih0 * ar * pybamm.sinh(i_coef * eta_H(data))

        def i_L(data):
            """
            Low plateau current [A] as defined by equation (5b) in [1]

            Input
            -------
            data : tuple
                Contains species S8, S4, S2, S, Sp and voltage V. Each species term and
                the voltage can be a scalar or array.

            Output
            -------
                Low plateau current [A] : pybamm.Variable

            """

            S8, S4, S2, S, Sp, V = data

            return -2 * il0 * ar * pybamm.sinh(i_coef * eta_L(data))

        def cth(data):
            """
            Theoretical capacity [Ah] of the cell as defined by equation (2) in [2].

            Input
            -------
            data : tuple
                Contains species S8, S4, S2, S, Sp and voltage V. Each species term and
                the voltage can be a scalar or array.

            Output
            -------
            Theoretical capacity [Ah] : pybamm.Variable

            """

            S8, S4, S2, S, Sp, V = data

            return (3 * ne * F * S8 / (ns8 * Ms) + ne * F * S4 / (ns4 * Ms)) / 3600

        def k_s_func(data):

            # represent the step function by a steep logistic curve
            return k_s_discharge + (k_s_charge - k_s_discharge) / (
                1 + pybamm.exp(-10 * (i_H(data) + i_L(data)))
            )

        ###################################
        # Dynamic model functions
        ###################################

        def algebraic_condition_func(S8, S4, S2, S, Sp, V):
            """
            Algebraic constraint on currents as defined by equation (7) in [1]

            Input
            -------
            S8, S4, S2, S, Sp, V : pybamm.Variable
                Sulfur species [g] and Terminal voltage [V]

            Output
            -------
            pybamm.Variable
                Should equal to zero if condition is satisfied.

            """

            # pack data list
            data = S8, S4, S2, S, Sp, V

            return i_H(data) + i_L(data) - I

        # RHS of ODE functions
        def f8(S8, S4, S2, S, Sp, V):
            """
            RHS of differential equation (8a) in [1]

            Input
            -------
            S8, S4, S2, S, Sp, V : pybamm.Variable
                Sulfur species [g] and Terminal voltage [V].
                Scalar or array dependent on species/voltage type.

            Output
            -------
            pybamm.Variable
                Scalar or array dependent on species/voltage type.

            """

            # pack data list
            data = S8, S4, S2, S, Sp, V

            return -(ns8 * Ms * i_H(data) / (ne * F)) - k_s_func(data) * S8

        def f4(S8, S4, S2, S, Sp, V):
            """
            RHS of differential equation (8b) in [1]

            Inputs
            --------
            S8, S4, S2, S, Sp, V : pybamm.Variable
                Sulfur species [g] and Terminal voltage [V].
                Scalar or array dependent on species/voltage type.

            Output
            -------
            pybamm.Variable
                Scalar or array dependent on species/voltage type.

            """

            # pack data list
            data = S8, S4, S2, S, Sp, V

            return (
                (ns8 * Ms * i_H(data) / (ne * F))
                + k_s_func(data) * S8
                - (ns4 * Ms * i_L(data) / (ne * F))
            )

        def f2(S8, S4, S2, S, Sp, V):
            """
            RHS of differential equation (8c) in [1]

            Inputs
            -------
            S8, S4, S2, S, Sp, V : pybamm.Variable
                Sulfur species [g] and Terminal voltage [V].
                Scalar or array dependent on species/voltage type.

            Output
            -------
            pybamm.Variable
                Scalar or array dependent on species/voltage type.

            """

            # pack data list
            data = S8, S4, S2, S, Sp, V

            return ns2 * Ms * i_L(data) / (ne * F)

        def f(S8, S4, S2, S, Sp, V):
            """
            RHS of differential equation (8d) in [1]

            Inputs
            -------
            S8, S4, S2, S, Sp, V : pybamm.Variable
                Sulfur species [g] and Terminal voltage [V].
                Scalar or array dependent on species/voltage type.

            Output
            -------
            pybamm.Variable
                Scalar or array dependent on species/voltage type.

            """

            # pack data list
            data = S8, S4, S2, S, Sp, V

            return (2 * ns * Ms * i_L(data) / (ne * F)) - k_p * Sp * (S - S_star) / (
                v * rho_s
            )

        def fp(S8, S4, S2, S, Sp, V):
            """
            RHS of differential equation (8e) in [1]

            Inputs
            -------
            S8, S4, S2, S, Sp, V : pybamm.Variable
                Sulfur species [g] and Terminal voltage [V].
                Scalar or array dependent on species/voltage type.

            Output
            -------
            pybamm.Variable
                Scalar or array dependent on species/voltage type.

            """

            return k_p * Sp * (S - S_star) / (v * rho_s)

        ##############################
        # model variables
        #############################

        # dynamic variables
        S8 = pybamm.Variable("S8 [g]")
        S4 = pybamm.Variable("S4 [g]")
        S2 = pybamm.Variable("S2 [g]")
        S = pybamm.Variable("S [g]")
        Sp = pybamm.Variable("Precipitated Sulfur [g]")
        alg = pybamm.Variable("Algebraic condition [-]")
        k_s = pybamm.Variable("Shuttle coefficient [s-1]")
        Sr = pybamm.Variable("Shuttle rate [g-1.s-1]")

        self.variables.update(
            {
                "S8 [g]": S8,
                "S4 [g]": S4,
                "S2 [g]": S2,
                "S [g]": S,
                "Precipitated Sulfur [g]": Sp,
                "Shuttle coefficient [s-1]": k_s_func((S8, S4, S2, S, Sp, V)),
                "Shuttle rate [g-1.s-1]": k_s_func((S8, S4, S2, S, Sp, V)) * S8,
                "High plateau potential [V]": E_H((S8, S4, S2, S, Sp, V)),
                "Low plateau potential [V]": E_L((S8, S4, S2, S, Sp, V)),
                "High plateau over-potential [V]": eta_H((S8, S4, S2, S, Sp, V)),
                "Low plateau over-potential [V]": eta_L((S8, S4, S2, S, Sp, V)),
                "High plateau current [A]": i_H((S8, S4, S2, S, Sp, V)),
                "Low plateau current [A]": i_L((S8, S4, S2, S, Sp, V)),
                "Theoretical capacity [Ah]": cth((S8, S4, S2, S, Sp, V)),
                "Algebraic condition": algebraic_condition_func(S8, S4, S2, S, Sp, V),
            }
        )

        #####################################
        # Dynamic model equations
        #####################################

        # ODEs
        dS8dt = f8(S8, S4, S2, S, Sp, V)
        dS4dt = f4(S8, S4, S2, S, Sp, V)
        dS2dt = f2(S8, S4, S2, S, Sp, V)
        dSpdt = fp(S8, S4, S2, S, Sp, V)
        dSdt = f(S8, S4, S2, S, Sp, V)

        self.rhs = {S8: dS8dt, S4: dS4dt, S2: dS2dt, S: dSdt, Sp: dSpdt}

        # Algebraic Condition
        algebraic_condition = algebraic_condition_func(S8, S4, S2, S, Sp, V)

        self.algebraic.update({V: algebraic_condition})

        ########################
        # Model events
        ########################

        CellQ = (
            m_s * 12 / 8 * F / Ms * 1 / 3600
        )  # cell capacity in Ah. used to calculate approx discharge/charge duration

        def voltage_event(V, V_max, V_min, I):
            if I > 0:
                output = V - V_min
            else:
                output = V_max - V

            return output

        def species_event(Sp, S4, m_s, I):
            if I > 0:
                output = Sp - 0.49869 * m_s
            else:
                output = S4 - 1e-6

            return output

        """
        self.events.extend(
            [
                pybamm.Event(
                    "Maximum voltage", voltage_event(V,V_max,V_min,I), pybamm.EventType.TERMINATION
                ),
                pybamm.Event(
                    "Species bounds",  species_event(Sp, S4, m_s, I),  pybamm.EventType.TERMINATION
                )
            ]
        )
        """

        ######################################
        # Discharge initial condition
        ######################################
        """
        Sets the initial species and voltage for discharge

        The values are found by considering the zero-current
        state of the battery. Set S8, S4, and Sp as written
        below. Then, solve eta_H = V, eta_L = V, the algebraic
        condition, and mass conservation for the remaining values.

        """
        S8_initial = 2.6946000000000003
        S4_initial = 0.0027
        S2_initial = 0.002697299116926997
        S_initial = 8.83072852310722e-10
        Sp_initial = 2.7e-06
        V_initial = 2.430277479547109

        self.initial_conditions.update(
            {
                self.variables["S8 [g]"]: pybamm.Scalar(S8_initial),
                self.variables["S4 [g]"]: pybamm.Scalar(S4_initial),
                self.variables["S2 [g]"]: pybamm.Scalar(S2_initial),
                self.variables["S [g]"]: pybamm.Scalar(S_initial),
                self.variables["Precipitated Sulfur [g]"]: pybamm.Scalar(Sp_initial),
                self.variables["Terminal voltage [V]"]: pybamm.Scalar(V_initial),
            }
        )

    ####################################
    # Extra functions for user
    ####################################
    def set_charge_initial_condition(self):
        """
        Sets the initial species and voltage condition for charge

        These conditions are found by discharging as 1A from
        the initial discharge conditions as found in
        self.set_discharge_initial_condition

        """
        S8_initial = 7.701508424389238e-12
        S4_initial = 0.004128610073594744
        S2_initial = 1.3492829940762894
        S_initial = 0.00012808077128429035
        Sp_initial = 1.3464603150711534
        V_initial = 2.254181286879344
        I_initial = -1

        self.initial_conditions = {
            self.variables["S8 [g]"]: pybamm.Scalar(S8_initial),
            self.variables["S4 [g]"]: pybamm.Scalar(S4_initial),
            self.variables["S2 [g]"]: pybamm.Scalar(S2_initial),
            self.variables["S [g]"]: pybamm.Scalar(S_initial),
            self.variables["Precipitated Sulfur [g]"]: pybamm.Scalar(Sp_initial),
            self.variables["Terminal voltage [V]"]: pybamm.Scalar(V_initial),
            self.variables["Current [A]"]: pybamm.Scalar(I_initial),
        }

    def set_initial_condition(self, init_cond):
        """Set initial conditions from init_cond dictionary."""
        self.initial_conditions = {
            self.variables[key]: pybamm.Scalar(init_cond[key])
            for key in init_cond.keys()
        }

    @property
    def default_parameter_values(self):

        return pybamm.ParameterValues(
            {
                "Temperature [K]": pybamm.Scalar(298),
                "Molar mass of S8 [g.mol-1]": pybamm.Scalar(32),
                "Number of S atoms in S [atoms]": pybamm.Scalar(1),
                "Number of S atoms in S2 [atoms]": pybamm.Scalar(2),
                "Number of S atoms in S4 [atoms]": pybamm.Scalar(4),
                "Number of S atoms in S8 [atoms]": pybamm.Scalar(8),
                "Electron number per reaction [electrons]": pybamm.Scalar(4),
                "Exchange current density H [A.m-2]": pybamm.Scalar(10),
                "Exchange current density L [A.m-2]": pybamm.Scalar(5),
                "Active reaction area per cell [m2]": pybamm.Scalar(0.960),
                "Mass of active sulfur per cell [g]": pybamm.Scalar(2.7),
                "Density of precipitated Sulfur [g.L-1]": pybamm.Scalar(2 * (10 ** 3)),
                "Electrolyte volume per cell [L]": pybamm.Scalar(0.0114),
                "Lower voltage cutoff [V]": pybamm.Scalar(2.15),
                "Upper voltage cutoff [V]": pybamm.Scalar(2.5),
                "Standard Potential H [V]": pybamm.Scalar(2.35),
                "Standard Potential L [V]": pybamm.Scalar(2.195),
                "Precipitation rate [s-1]": pybamm.Scalar(100),
                "S saturation mass [g]": pybamm.Scalar(0.0001),
                "Shuttle rate coefficient during charge [s-1]": pybamm.Scalar(0.0002),
                "Shuttle rate coefficient during discharge [s-1]": pybamm.Scalar(0),
                "Current [A]": pybamm.Scalar(1),
                "Number of cells connected in series to make a battery": pybamm.Scalar(
                    1
                ),
            }
        )

    def new_copy(self, options=None):
        "Create an empty copy with identical options, or new options if specified"
        options = options or self.options
        new_model = self.__class__(options=options, name=self.name)
        new_model.use_jacobian = self.use_jacobian
        new_model.use_simplify = self.use_simplify
        new_model.convert_to_format = self.convert_to_format
        new_model.timescale = self.timescale
        new_model.length_scales = self.length_scales
        return new_model
