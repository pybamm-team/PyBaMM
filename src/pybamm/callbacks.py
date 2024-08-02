import pybamm
import numpy as np
import inspect


def setup_callbacks(callbacks):
    callbacks = callbacks or []
    if not isinstance(callbacks, list):
        callbacks = [callbacks]

    # Check if there is a logging callback already, if not add the default one
    has_logging_callback = any(isinstance(cb, LoggingCallback) for cb in callbacks)
    if not has_logging_callback:
        callbacks.append(LoggingCallback())

    return CallbackList(callbacks)


class Callback:
    """
    Base class for callbacks, for documenting callback methods.

    Callbacks are used to perform actions (e.g. logging, saving) at certain points in
    the simulation. Each callback method is named `on_<event>`, where `<event>`
    describes the point at which the callback is called. For example, the callback
    `on_experiment_start` is called at the start of an experiment simulation. In
    general, callbacks take a single argument, `logs`, which is a dictionary of
    information about the simulation. Each callback method should return `None`
    (the output of the method is ignored).

    **EXPERIMENTAL** - this class is experimental and the callback interface may
    change in future releases.
    """

    def on_experiment_start(self, logs):
        """
        Called at the start of an experiment simulation.
        """
        pass  # pragma: no cover

    def on_cycle_start(self, logs):
        """
        Called at the start of each cycle in an experiment simulation.
        """
        pass  # pragma: no cover

    def on_step_start(self, logs):
        """
        Called at the start of each step in an experiment simulation.
        """
        pass  # pragma: no cover

    def on_step_end(self, logs):
        """
        Called at the end of each step in an experiment simulation.
        """
        pass  # pragma: no cover

    def on_cycle_end(self, logs):
        """
        Called at the end of each cycle in an experiment simulation.
        """
        pass  # pragma: no cover

    def on_experiment_end(self, logs):
        """
        Called at the end of an experiment simulation.
        """
        pass  # pragma: no cover

    def on_experiment_error(self, logs):
        """
        Called when a SolverError occurs during an experiment simulation.

        For example, this could be used to send an error alert with a bug report when
        running batch simulations in the cloud.
        """
        pass  # pragma: no cover

    def on_experiment_infeasible_time(self, logs):
        """
        Called when an experiment simulation is infeasible due to reaching maximum time.
        """
        pass  # pragma: no cover

    def on_experiment_infeasible_event(self, logs):
        """
        Called when an experiment simulation is infeasible due to an event.
        """
        pass  # pragma: no cover


########################################################################################
class CallbackList(Callback):
    """
    Container abstracting a list of callbacks, so that they can be called in a
    single step e.g. `callbacks.on_simulation_end(...)`.

    This is done without having to redefine the method each time by using the
    `callback_loop_decorator` decorator, which is applied to every method that starts
    with `on_`, using the `inspect` module.

    If better control over how the callbacks are called is required, it might be better
    to be more explicit with the for loop.
    """

    def __init__(self, callbacks):
        self.callbacks = callbacks

    def __len__(self):
        return len(self.callbacks)

    def __getitem__(self, index):
        return self.callbacks[index]


def callback_loop_decorator(func):
    """
    A decorator to call the function on every callback in `self.callbacks`
    """

    def wrapper(self, *args, **kwargs):
        for callback in self.callbacks:
            # call the function on the callback
            getattr(callback, func.__name__)(*args, **kwargs)

    return wrapper


# inspect.getmembers finds all the methods in the Callback class
for name, func in inspect.getmembers(CallbackList, inspect.isfunction):
    if name.startswith("on_"):
        # Replaces each function with the decorated version
        setattr(CallbackList, name, callback_loop_decorator(func))

########################################################################################


class LoggingCallback(Callback):
    """
    Logging callback, implements methods to log progress of the simulation.

    Parameters
    ----------
    logfile : str, optional
        Where to send the log output. If None, uses pybamm's logger.
    """

    def __init__(self, logfile=None):
        self.logfile = logfile
        if logfile is None:
            # Use pybamm's logger, which prints to command line
            self.logger = pybamm.logger
        else:
            # Use a custom logger, this will have its own level so set it to the same
            # level as the pybamm logger (users can override this)
            self.logger = pybamm.get_new_logger(__name__, logfile)
            self.logger.setLevel(pybamm.logger.level)

    def on_experiment_start(self, logs):
        # Clear the log file
        self.logger.info("Start running experiment")
        if self.logfile is not None:
            with open(self.logfile, "w") as f:
                f.write("")

    def on_cycle_start(self, logs):
        cycle_num, num_cycles = logs["cycle number"]
        total_time = logs["elapsed time"]
        self.logger.notice(
            f"Cycle {cycle_num}/{num_cycles} ({total_time} elapsed) " + "-" * 20
        )

    def on_step_start(self, logs):
        cycle_num, num_cycles = logs["cycle number"]
        step_num, cycle_length = logs["step number"]
        operating_conditions = logs["step operating conditions"]
        self.logger.notice(
            f"Cycle {cycle_num}/{num_cycles}, step {step_num}/{cycle_length}: "
            f"{operating_conditions}"
        )

    def on_step_end(self, logs):
        time_stop = logs["stopping conditions"]["time"]
        if time_stop is not None:
            time_now = logs["experiment time"]
            if time_now < time_stop:
                self.logger.notice(
                    f"Time is now {time_now:.3f} s, will stop at {time_stop:.3f} s."
                )
            else:
                self.logger.notice(
                    f"Stopping experiment since time ({time_now:.3f} s) "
                    f"has reached stopping time ({time_stop:.3f} s)."
                )

    def on_cycle_end(self, logs):
        cap_stop = logs["stopping conditions"]["capacity"]
        if cap_stop is not None:
            cap_now = logs["summary variables"]["Capacity [A.h]"]
            cap_start = logs["start capacity"]
            if np.isnan(cap_now) or cap_now > cap_stop:
                self.logger.notice(
                    f"Capacity is now {cap_now:.3f} Ah (originally {cap_start:.3f} Ah, "
                    f"will stop at {cap_stop:.3f} Ah)"
                )
            else:
                self.logger.notice(
                    f"Stopping experiment since capacity ({cap_now:.3f} Ah) "
                    f"is below stopping capacity ({cap_stop:.3f} Ah)."
                )

        voltage_stop = logs["stopping conditions"]["voltage"]
        if voltage_stop is not None:
            min_voltage = logs["summary variables"]["Minimum voltage [V]"]
            if min_voltage > voltage_stop[0]:
                self.logger.notice(
                    f"Minimum voltage is now {min_voltage:.3f} V "
                    f"(will stop at {voltage_stop[0]:.3f} V)"
                )
            else:
                self.logger.notice(
                    f"Stopping experiment since minimum voltage ({min_voltage:.3f} V) "
                    f"is below stopping voltage ({voltage_stop[0]:.3f} V)."
                )

    def on_experiment_end(self, logs):
        elapsed_time = logs["elapsed time"]
        self.logger.notice(f"Finish experiment simulation, took {elapsed_time}")

    def on_experiment_error(self, logs):
        error = logs["error"]
        pybamm.logger.error(f"Simulation error: {error}")

    def on_experiment_infeasible_time(self, logs):
        duration = logs["step duration"]
        cycle_num = logs["cycle number"][0]
        step_num = logs["step number"][0]
        operating_conditions = logs["step operating conditions"]
        self.logger.warning(
            f"\n\n\tExperiment is infeasible: default duration ({duration} seconds) "
            f"was reached during '{operating_conditions}'. The returned solution only "
            f"contains up to step {step_num} of cycle {cycle_num}. "
            "Please specify a duration in the step instructions."
        )

    def on_experiment_infeasible_event(self, logs):
        termination = logs["termination"]
        cycle_num = logs["cycle number"][0]
        step_num = logs["step number"][0]
        operating_conditions = logs["step operating conditions"]
        self.logger.warning(
            f"\n\n\tExperiment is infeasible: '{termination}' was "
            f"triggered during '{operating_conditions}'. The returned solution only "
            f"contains up to step {step_num} of cycle {cycle_num}. "
        )
