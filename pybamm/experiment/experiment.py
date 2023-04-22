#
# Experiment class
#

import pybamm


class Experiment:
    """
    Base class for experimental conditions under which to run the model. In general, a
    list of operating conditions should be passed in. Each operating condition should
    be of the form "Do this for this long" or "Do this until this happens". For example,
    "Charge at 1 C for 1 hour", or "Charge at 1 C until 4.2 V", or "Charge at 1 C for 1
    hour or until 4.2 V at 25oC". The instructions can be of the form
    "(Dis)charge at x A/C/W", "Rest", or "Hold at x V until y A at z oC". The running
    time should be a time in seconds, minutes or
    hours, e.g. "10 seconds", "3 minutes" or "1 hour". The stopping conditions should be
    a circuit state, e.g. "1 A", "C/50" or "3 V". The parameter drive_cycles is
    mandatory to run drive cycle. For example, "Run x", then x must be the key
    of drive_cycles dictionary. The temperature should be provided after the stopping
    condition but before the period, e.g. "1 A at 25 oC (1 second period)". It is
    not essential to provide a temperature and a global temperature can be set either
    from within the paramter values of passing a temperature to this experiment class.
    If the temperature is not specified in a line, then the global temperature is used,
    even if another temperature has been set in an earlier line.

    Parameters
    ----------
    operating_conditions : list
        List of operating conditions
    period : string, optional
        Period (1/frequency) at which to record outputs. Default is 1 minute. Can be
        overwritten by individual operating conditions.
    temperature: float, optional
        The ambient air temperature in degrees Celsius at which to run the experiment.
        Default is None whereby the ambient temperature is taken from the parameter set.
        This value is overwritten if the temperature is specified in a step.
    termination : list, optional
        List of conditions under which to terminate the experiment. Default is None.
    drive_cycles : dict
        Dictionary of drive cycles to use for this experiment.
    """

    def __init__(
        self,
        operating_conditions,
        period="1 minute",
        temperature=None,
        termination=None,
        drive_cycles=None,
        cccv_handling=None,
    ):
        if cccv_handling is not None:
            raise ValueError(
                "cccv_handling has been deprecated, use "
                "`pybamm.experiment.cccv_ode(current, voltage)` instead to produce the "
                "same behavior as the old `cccv_handling='ode'`"
            )
        if drive_cycles is None:
            raise ValueError(
                "drive_cycles should now be passed as an experiment step object, e.g. "
                "`pybamm.experiment.current(drive_cycle)`"
            )
        # Save arguments for copying
        self.args = (
            operating_conditions,
            period,
            temperature,
            termination,
        )

        self.period = self.convert_time_to_seconds(period.split())
        self.temperature = temperature

        operating_conditions_cycles = []
        for cycle in operating_conditions:
            # Check types and convert to list
            if not isinstance(cycle, tuple):
                cycle = [cycle]
            # Convert strings to pybamm.experiment._Step objects
            processed_cycle = []
            for step in cycle:
                if isinstance(step, str):
                    processed_cycle.append(pybamm.experiment.string(step))
                else:
                    processed_cycle.append(step)
            operating_conditions_cycles.append(tuple(processed_cycle))

        self.operating_conditions_cycles = operating_conditions_cycles
        self.cycle_lengths = [len(cycle) for cycle in operating_conditions_cycles]

        operating_conditions_steps = [
            cond for cycle in operating_conditions_cycles for cond in cycle
        ]
        self.operating_conditions = operating_conditions_steps

        self.termination_string = termination
        self.termination = self.read_termination(termination)

    def __str__(self):
        return str(self.operating_conditions_cycles)

    def copy(self):
        return Experiment(*self.args)

    def __repr__(self):
        return "pybamm.Experiment({!s})".format(self)

    def read_termination(self, termination):
        """
        Read the termination reason. If this condition is hit, the experiment will stop.
        """
        if termination is None:
            return {}
        elif isinstance(termination, str):
            termination = [termination]

        termination_dict = {}
        for term in termination:
            term_list = term.split()
            if term_list[-1] == "capacity":
                end_discharge = "".join(term_list[:-1])
                end_discharge.replace("A.h", "Ah")
                if end_discharge.endswith("%"):
                    end_discharge_percent = end_discharge.split("%")[0]
                    termination_dict["capacity"] = (float(end_discharge_percent), "%")
                elif end_discharge.endswith("Ah"):
                    end_discharge_Ah = end_discharge.split("Ah")[0]
                    termination_dict["capacity"] = (float(end_discharge_Ah), "Ah")
                else:
                    raise ValueError(
                        "Capacity termination must be given in the form "
                        "'80%', '4Ah', or '4A.h'"
                    )
            elif term.endswith("V"):
                end_discharge_V = term.split("V")[0]
                termination_dict["voltage"] = (float(end_discharge_V), "V")
            else:
                raise ValueError(
                    "Only capacity or voltage can be provided as a termination reason, "
                    "e.g. '80% capacity', '4 Ah capacity', or '2.5 V'"
                )
        return termination_dict
