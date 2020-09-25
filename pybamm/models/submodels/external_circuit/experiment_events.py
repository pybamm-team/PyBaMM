#
# Model to impose the events for experiments
#
import pybamm


class ExperimentEvents(pybamm.BaseSubModel):
    """Model to impose the events for experiments."""

    def __init__(self, param):
        super().__init__(param)

    def set_events(self, variables):
        # add current and voltage events to the model
        # current events both negative and positive to catch specification
        n_cells = self.param.n_cells
        self.events.extend(
            [
                pybamm.Event(
                    "Current cut-off (positive) [A] [experiment]",
                    variables["Current [A]"]
                    - abs(pybamm.InputParameter("Current cut-off [A]")),
                ),
                pybamm.Event(
                    "Current cut-off (negative) [A] [experiment]",
                    variables["Current [A]"]
                    + abs(pybamm.InputParameter("Current cut-off [A]")),
                ),
                pybamm.Event(
                    "Voltage cut-off [V] [experiment]",
                    variables["Terminal voltage [V]"]
                    - pybamm.InputParameter("Voltage cut-off [V]") / n_cells,
                ),
            ]
        )
