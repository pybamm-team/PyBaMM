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

    def build(self):
        number_of_elements = self.model_options["number of rc elements"] + 1
        ocv = self.coupled_variable("Open-circuit voltage [V]")
        diffusion_overpotential = self.coupled_variable("Diffusion overpotential [V]")
        current = self.coupled_variable("Current [A]")
        overpotential = pybamm.Scalar(0)
        for i in range(number_of_elements):
            overpotential += self.coupled_variable(f"Element-{i} overpotential [V]")

        voltage = ocv + overpotential + diffusion_overpotential

        def x_not_zero(x):
            return ((x > 0) + (x < 0)) * x + (x >= 0) * (x <= 0)

        non_zero_current = x_not_zero(current)

        self.variables.update(
            {
                "Voltage [V]": voltage,
                "Overpotential [V]": overpotential,
                "Battery voltage [V]": voltage,
                "Power [W]": voltage * current,
                "Resistance [Ohm]": pybamm.sign(current) * voltage / non_zero_current,
            }
        )

        # Cut-off voltage for event switch with casadi 'fast with events'
        tol = 0.125
        self.events += [
            pybamm.Event(
                "Maximum voltage [V]",
                self.param.voltage_high_cut - voltage,
                pybamm.EventType.TERMINATION,
            ),
            pybamm.Event(
                "Minimum voltage [V]",
                voltage - self.param.voltage_low_cut,
                pybamm.EventType.TERMINATION,
            ),
            pybamm.Event(
                "Minimum voltage switch",
                voltage - (self.param.voltage_low_cut - tol),
                pybamm.EventType.SWITCH,
            ),
            pybamm.Event(
                "Maximum voltage switch",
                voltage - (self.param.voltage_high_cut + tol),
                pybamm.EventType.SWITCH,
            ),
        ]
