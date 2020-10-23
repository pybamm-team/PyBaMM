#
# Marinescu et al (2016) Li-S model
#
import pybamm
from .base_lithium_sulfur_model import BaseModel


class MarinescuEtAl2016(BaseModel):
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

    .. [2]  Marinescu, M., O’Neill, L., Zhang, T., Walus, S., Wilson, T. E., &
            Offer, G. J. (2018).
            Irreversible vs reversible capacity fade of lithium-sulfur batteries
            during cycling: the effects of precipitation and shuttle. Journal of
            The Electrochemical Society, 165(1), A6107-A6118.
    """

    def __init__(self, options=None, name="Marinescu et al. (2016) model"):
        super().__init__(options, name)

        # Note: you might want to move the setting of how the operating mode
        # set up to the base model as it will be general to all models. I'll
        # leave it here for now, but as a function
        self.set_external_circuit_submodel()

        # Get V and I since they have been set in a separate function
        V = self.variables["Terminal voltage [V]"]
        I = self.variables["Current [A]"]

        # Model parameters as defined in table (1) in [1]. Parameters with 'H'
        # or 'L' in the name represent the high and low plateau parameter, respectively.

        # Note: here I'm just pointing your params to the standard parameters for
        # lithium sulfur. You could just call e.g. self.param.R directly when you
        # need it, but sometimes it is convient to call R = self.param.R to avoid
        # lots of typing
        param = self.param

        # standard parameters
        R = param.R
        F = param.F
        T = param.T_ref

        # model-specific known parameters
        Ms = param.Ms
        ns = param.ns
        ns2 = param.ns2
        ns4 = param.ns4
        ns8 = param.ns8
        ne = param.ne
        ih0 = param.ih0
        il0 = param.il0
        m_s = param.m_s
        rho_s = param.rho_s
        EH0 = param.EH0
        EL0 = param.EL0

        # model-specific unknown parameters
        v = param.v
        ar = param.ar
        k_p = param.k_p
        S_star = param.S_star
        k_s_charge = param.k_s_charge
        k_s_discharge = param.k_s_discharge

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

        # Note: the string names you give variables should probably be more
        # informative
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
                "Algebraic condition [-]": algebraic_condition_func(
                    S8, S4, S2, S, Sp, V
                ),
            }
        )

        #####################################
        # Dynamic model equations
        #####################################

        # Note: I think it would be easier to read if you just defined the rhs
        # and algebraic directly here rather than using the functins defined
        # further up (I think they are only called here?), e.g.:
        # ds8dt = -(ns8 * Ms * i_H(data) / (ne * F)) - k_s_func(data) * S8

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

        CellQ = (
            m_s * 12 / 8 * F / Ms * 1 / 3600
        )  # cell capacity in Ah. used to calculate approx discharge/charge duration

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
        self.initial_conditions.update(
            {
                self.variables["S8 [g]"]: param.S8_initial,
                self.variables["S4 [g]"]: param.S4_initial,
                self.variables["S2 [g]"]: param.S2_initial,
                self.variables["S [g]"]: param.S_initial,
                self.variables["Precipitated Sulfur [g]"]: param.Sp_initial,
                self.variables["Terminal voltage [V]"]: param.V_initial,
            }
        )

    def set_external_circuit_submodel(self):
        """
        Define how the external circuit defines the boundary conditions for the model,
        e.g. (not necessarily constant-) current, voltage, etc
        """

        # Set variable for the terminal voltage
        V = pybamm.Variable("Terminal voltage [V]")
        self.variables.update({"Terminal voltage [V]": V})

        # Set up current operated. Here the current is just provided as a
        # parameter
        if self.options["operating mode"] == "current":
            I = pybamm.Parameter("Current function [A]")
            self.variables.update({"Current [A]": I})

        # If the the operating mode of the simulation is "with experiment" the
        # model option "operating mode" is the callable function
        # 'constant_current_constant_voltage_constant_power'
        elif callable(self.options["operating mode"]):
            # For the experiment we solve an extra algebraic equation
            # to determine the current
            I = pybamm.Variable("Current [A]")
            self.variables.update({"Current [A]": I})
            control_function = self.options["operating mode"]

            # 'constant_current_constant_voltage_constant_power' is a function
            # of current and voltage via the variables dict (see pybamm.Simulation)
            self.algebraic = {I: control_function(self.variables)}
            self.initial_conditions[I] = pybamm.Parameter("Current function [A]")
