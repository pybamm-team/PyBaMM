#
# Base class for callbacks and some useful callbacks for pybamm
# Callbacks are used to perform actions (e.g. logging, saving)
# at certain points in the simulation
# Inspired by Keras callbacks
# https://github.com/keras-team/keras/blob/master/keras/callbacks.py
#
import pybamm
import logging
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
        pass

    def on_cycle_start(self, logs):
        """
        Called at the start of each cycle in an experiment simulation.
        """
        pass

    def on_step_start(self, logs):
        """
        Called at the start of each step in an experiment simulation.
        """
        pass

    def on_step_end(self, logs):
        """
        Called at the end of each step in an experiment simulation.
        """
        pass

    def on_cycle_end(self, logs):
        """
        Called at the end of each cycle in an experiment simulation.
        """
        pass

    def on_experiment_end(self, logs):
        """
        Called at the end of an experiment simulation.
        """
        pass

    def on_experiment_error(self, logs):
        """
        Called when a SolverError occurs during an experiment simulation.

        For example, this could be used to send an error alert with a bug report when
        running batch simulations in the cloud.
        """
        pass

    def on_experiment_infeasible(self, logs):
        """
        Called when an experiment simulation is infeasible.
        """
        pass


class CallbackList(Callback):
    """
    Container abstracting a list of callbacks, so that they can be called in a
    single step e.g. `callbacks.on_simulation_end(...)`.

    This is done without having to redefine the method each time by using the
    `callback_loop_decorator` decorator, which is applied to every method that starts
    with `on_`, using the `inspect` module.
    See https://stackoverflow.com/questions/1367514/how-to-decorate-a-method-inside-a-class.

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


class LoggingCallback(Callback):
    """
    Logging callback, implements methods to log progress of the simulation.

    Parameters
    ----------
    log_output : str, optional
        Where to send the log output. If None, uses pybamm's logger.

    **Extends:** :class:`pybamm.callbacks.Callback`
    """

    def __init__(self, log_output=None):
        self.log_output = log_output
        if log_output is None:
            # Use pybamm's logger, which prints to command line
            self.logger = pybamm.logger
        else:
            # Use a custom logger, this will have its own level so set it to the same
            # level as the pybamm logger (users can override this)
            self.logger = pybamm.get_new_logger(__name__, log_output)
            self.logger.setLevel(pybamm.logger.level)

    def on_experiment_start(self, logs):
        # Clear the log file
        if self.log_output is not None:
            with open(self.log_output, "w") as f:
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
        operating_conditions = logs["operating conditions"]
        self.logger.notice(
            f"Cycle {cycle_num}/{num_cycles}, step {step_num}/{cycle_length}: "
            f"{operating_conditions}"
        )

    def on_step_end(self, logs):
        pass

    def on_cycle_end(self, logs):
        pass

    def on_experiment_end(self, logs):
        elapsed_time = logs["elapsed time"]
        self.logger.notice("Finish experiment simulation, took {}".format(elapsed_time))

    def on_experiment_error(self, logs):
        pass

    def on_experiment_infeasible(self, logs):
        termination = logs["termination"]
        cycle_num = logs["cycle number"][0]
        step_num = logs["step number"][0]
        operating_conditions = logs["operating conditions"]
        self.logger.warning(
            f"\n\n\tExperiment is infeasible: '{termination}' was "
            f"triggered during '{operating_conditions}'. The returned solution only "
            f"contains the first {cycle_num-1} cycles, up to step {step_num-1}. "
            "Try reducing the current, shortening the time interval, or reducing the "
            "period.\n\n"
        )


class SaveIfErrorCallback(Callback):
    """
    Callback to save the simulation configuration to disk if an error occurs.

    Parameters
    ----------
    filename : str
        The filename to save the model to.

    **Extends:** :class:`pybamm.callbacks.Callback`
    """

    def __init__(self, filename):
        self.filename = filename

    def on_experiment_error(self, logs):
        pybamm.save(logs["model"], self.filename)
