#
# Base class for callbacks and some useful callbacks for pybamm
# Callbacks are used to perform actions (e.g. logging, saving)
# at certain points in the simulation
# Inspired by Keras callbacks
# https://github.com/keras-team/keras/blob/master/keras/callbacks.py
#
import abc


def setup_callbacks(callbacks):
    callbacks = callbacks or []
    if not isinstance(callbacks, list):
        callbacks = [callbacks]

    return CallbackList(callbacks)


class Callback(metaclass=abc.ABCMeta):
    """
    Abstract base class for callbacks, for documenting callback methods.

    Callbacks are used to perform actions (e.g. logging, saving) at certain points in
    the simulation.

    **EXPERIMENTAL** - this class is experimental and the callback interface may
    change in future releases.
    """

    @abc.abstractmethod
    def on_simulation_end(self, sim):
        """
        Called at the end of the simulation.

        Parameters
        ----------
        sim : :class:`pybamm.Simulation`
            Simulation object to be logged.
        """


class CallbackList(Callback):
    """
    Container abstracting a list of callbacks, so that they can be called in a
    single step e.g. `callbacks.on_simulation_end(...)`
    """

    def __init__(self, callbacks):
        self.callbacks = callbacks
        # Could add some default callbacks to the list here

    def on_simulation_end(self, *args, **kwargs):
        for callback in self:
            callback.on_simulation_end(*args, **kwargs)

    def __iter__(self):
        """
        Magic function called by `for ... in callbacks`
        """
        return iter(self.callbacks)


class LoggingCallback(Callback):
    """
    Logging callback, implements methods to log progress.

    Parameters
    ----------
    logs : str
        Path to log file.

    **Extends:** :class:`pybamm.callbacks.Callback`
    """

    def __init__(self, logs):
        self.logs = logs

    def on_simulation_end(self, sim):
        with open(self.logs, "w") as f:
            print(sim.solution.termination, file=f)
