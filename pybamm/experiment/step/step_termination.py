import pybamm
import numpy as np


class BaseTermination:
    def __init__(self, value):
        self.value = value

    def get_event(self, variables, step_value):
        raise NotImplementedError


class CrateTermination(BaseTermination):
    def get_event(self, variables, step_value):
        event = pybamm.Event(
            "C-rate cut-off [A] [experiment]",
            abs(variables["C-rate"]) - self.value,
        )
        return event


class CurrentTermination(BaseTermination):
    def get_event(self, variables, step_value):
        event = pybamm.Event(
            "Current cut-off [A] [experiment]",
            abs(variables["Current [A]"]) - self.value,
        )
        return event


class VoltageTermination(BaseTermination):
    def get_event(self, variables, step_value):
        # The voltage event should be positive at the start of charge/
        # discharge. We use the sign of the current or power input to
        # figure out whether the voltage event is greater than the starting
        # voltage (charge) or less (discharge) and set the sign of the
        # event accordingly
        if isinstance(step_value, pybamm.Symbol):
            inpt = {"start time": 0}
            init_curr = step_value.evaluate(t=0, inputs=inpt).flatten()[0]
        else:
            init_curr = step_value
        sign = np.sign(init_curr)
        if sign > 0:
            name = "Discharge"
        else:
            name = "Charge"
        if sign != 0:
            # Event should be positive at initial conditions for both
            # charge and discharge
            event = pybamm.Event(
                f"{name} voltage cut-off [V] [experiment]",
                sign * (variables["Battery voltage [V]"] - self.value),
            )
            return event


class CustomTermination(BaseTermination):
    def __init__(self, name, event_function):
        self.name = name
        self.event_function = event_function

    def get_event(self, variables, step_value):
        return pybamm.Event(self.name, self.event_function(variables))


def read_termination(termination):
    if isinstance(termination, tuple):
        typ, value = termination
    else:
        return termination

    termination_class = {
        "current": CurrentTermination,
        "voltage": VoltageTermination,
        "C-rate": CrateTermination,
    }[typ]
    return termination_class(value)
