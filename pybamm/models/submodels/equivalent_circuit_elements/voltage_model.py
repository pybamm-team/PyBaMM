import pybamm


class VoltageModel(pybamm.BaseSubModel):
    """
    Voltage model for use with equivalent
    circuits. This model is used to calculate
    the voltage and total overpotentials
    from the other elements in the circuit.

    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel
    options : dict, optional
        A dictionary of options to be passed to the model.
    """

    def __init__(self, param, options=None):
        super().__init__(param)
        self.model_options = options

    def get_coupled_variables(self, variables):
        ocv = variables["Open-circuit voltage [V]"]

        number_of_rc_elements = self.model_options["number of rc elements"]
        number_of_elements = number_of_rc_elements + 1

        overpotential = pybamm.Scalar(0)
        for i in range(number_of_elements):
            overpotential += variables[f"Element-{i} overpotential [V]"]

        voltage = ocv + overpotential

        # Power and Resistance
        current = variables["Current [A]"]

        def x_not_zero(x):
            return ((x > 0) + (x < 0)) * x + (x >= 0) * (x <= 0)

        non_zero_current = x_not_zero(current)

        variables.update(
            {
                "Voltage [V]": voltage,
                "Overpotential [V]": overpotential,
                "Battery voltage [V]": voltage,
                "Power [W]": voltage * current,
                "Resistance [Ohm]": pybamm.sign(current) * voltage / non_zero_current,
            }
        )

        return variables

    def set_events(self, variables):
        voltage = variables["Voltage [V]"]

        # Add voltage events
        maximum_voltage = pybamm.Event(
            "Maximum voltage [V]",
            self.param.voltage_high_cut - voltage,
            pybamm.EventType.TERMINATION,
        )
        self.events.append(maximum_voltage)

        minimum_voltage = pybamm.Event(
            "Minimum voltage [V]",
            voltage - self.param.voltage_low_cut,
            pybamm.EventType.TERMINATION,
        )
        self.events.append(minimum_voltage)

        # Cut-off voltage for event switch with casadi 'fast with events'
        tol = 0.125
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
