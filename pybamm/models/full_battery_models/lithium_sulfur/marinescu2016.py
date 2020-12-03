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

        # set external variables
        self.set_external_circuit_submodel()
        V = self.variables["Terminal voltage [V]"]
        I = self.variables["Current [A]"]

        # set internal variables
        S8 = pybamm.Variable("S8 [g]")
        S4 = pybamm.Variable("S4 [g]")
        S2 = pybamm.Variable("S2 [g]")
        S = pybamm.Variable("S [g]")
        Sp = pybamm.Variable("Precipitated Sulfur [g]")

        #######################################
        # Model parameters as defined in table (1) in [1]. Parameters with 'H' or
        # 'L' in the name represent the high and low plateau parameter, respectively.
        #######################################
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

        i_coef = ne * F / (2 * R * T)
        E_H_coef = R * T / (4 * F)
        f_h = (ns4 ** 2) * Ms * v / ns8
        f_l = (ns ** 2) * ns2 * Ms ** 2 * (v ** 2) / ns4

        #######################################################
        # Non-dynamic model functions
        #######################################################

        # High plateau potenital [V] as defined by equation (2a) in [1]
        E_H = EH0 + E_H_coef * pybamm.log(f_h * S8 / (S4 ** 2))

        # Low plateau potenital [V] as defined by equation (2b) in [1]
        E_L = EL0 + E_H_coef * pybamm.log(f_l * S4 / (S2 * (S ** 2)))

        # High plateau over-potenital [V] as defined by equation (6a) in [1]
        eta_H = V - E_H

        # Low plateau over-potenital [V] as defined by equation (6b) in [1]
        eta_L = V - E_L

        # High plateau current [A] as defined by equation (5a) in [1]
        i_H = -2 * ih0 * ar * pybamm.sinh(i_coef * eta_H)

        # Low plateau current [A] as defined by equation (5b) in [1]
        i_L = -2 * il0 * ar * pybamm.sinh(i_coef * eta_L)

        # Theoretical capacity [Ah] of the cell as defined by equation (2) in [2]
        cth = (3 * ne * F * S8 / (ns8 * Ms) + ne * F * S4 / (ns4 * Ms)) / 3600

        # Shuttle coefficient
        k_s = k_s_charge * (I < 0) + k_s_discharge * (I >= 0)

        ###################################
        # Dynamic model functions
        ###################################

        # Algebraic constraint on currents as defined by equation (7) in [1]
        algebraic_condition = i_H + i_L - I
        self.algebraic.update({V: algebraic_condition})

        # Differential equation (8a) in [1]
        dS8dt = -(ns8 * Ms * i_H / (ne * F)) - k_s * S8

        # Differential equation (8b) in [1]
        dS4dt = (ns8 * Ms * i_H / (ne * F)) + k_s * S8 - (ns4 * Ms * i_L / (ne * F))

        # Differential equation (8c) in [1]
        dS2dt = ns2 * Ms * i_L / (ne * F)

        # Differential equation (8d) in [1]
        dSdt = (2 * ns * Ms * i_L / (ne * F)) - k_p * Sp * (S - S_star) / (v * rho_s)

        # Differential equation (8e) in [1]
        dSpdt = k_p * Sp * (S - S_star) / (v * rho_s)

        self.rhs.update({S8: dS8dt, S4: dS4dt, S2: dS2dt, S: dSdt, Sp: dSpdt})

        ##############################
        # Model variables
        #############################

        self.variables.update(
            {
                "Time [s]": pybamm.t * self.timescale,
                "S8 [g]": S8,
                "S4 [g]": S4,
                "S2 [g]": S2,
                "S [g]": S,
                "Precipitated Sulfur [g]": Sp,
                "Shuttle coefficient [s-1]": k_s,
                "Shuttle rate [g-1.s-1]": k_s * S8,
                "High plateau potential [V]": E_H,
                "Low plateau potential [V]": E_L,
                "High plateau over-potential [V]": eta_H,
                "Low plateau over-potential [V]": eta_L,
                "High plateau current [A]": i_H,
                "Low plateau current [A]": i_L,
                "Theoretical capacity [Ah]": cth,
                "Algebraic condition": algebraic_condition,
            }
        )

        ######################################
        # Discharge initial condition
        # The values are found by considering the zero-current
        # state of the battery. Set S8, S4, and Sp as written
        # below. Then, solve eta_H = V, eta_L = V, the algebraic
        # condition, and mass conservation for the remaining values.
        ######################################

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

        ######################################
        # Model events
        ######################################
        tol = 1e-4
        self.events.append(
            pybamm.Event(
                "Minimum voltage",
                V - self.param.voltage_low_cut,
                pybamm.EventType.TERMINATION,
            )
        )
        self.events.append(
            pybamm.Event(
                "Maximum voltage",
                V - self.param.voltage_high_cut,
                pybamm.EventType.TERMINATION,
            )
        )
        #self.events.append(
        #    pybamm.Event(
        #        "Zero theoretical capacity", cth - tol, pybamm.EventType.TERMINATION
        #    )
        #)

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

        # Add variable for discharge capacity
        Q = pybamm.Variable("Discharge capacity [A.h]")
        self.variables.update({"Discharge capacity [A.h]": Q})
        self.rhs.update({Q: I * self.param.timescale / 3600})
        self.initial_conditions.update({Q: pybamm.Scalar(0)})
