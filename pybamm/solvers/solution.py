#
# Solution class
#
import numbers
import numpy as np
import pickle
import pybamm
from collections import defaultdict


class Solution(object):
    """
    Class containing the solution of, and various attributes associated with, a PyBaMM
    model.

    Parameters
    ----------
    t : :class:`numpy.array`, size (n,)
        A one-dimensional array containing the times at which the solution is evaluated
    y : :class:`numpy.array`, size (m, n)
        A two-dimensional array containing the values of the solution. y[i, :] is the
        vector of solutions at time t[i].
    t_event : :class:`numpy.array`, size (1,)
        A zero-dimensional array containing the time at which the event happens.
    y_event : :class:`numpy.array`, size (m,)
        A one-dimensional array containing the value of the solution at the time when
        the event happens.
    termination : str
        String to indicate why the solution terminated

    """

    def __init__(self, t, y, t_event=None, y_event=None, termination="final time"):
        self.t = t
        self.y = y
        self.t_event = t_event
        self.y_event = y_event
        self.termination = termination
        # initialize empty inputs and model, to be populated later
        self.inputs = {}
        self._model = None

        # initiaize empty variables and data
        self._variables = {}
        self.data = {}

        # initialize empty known evals
        self.known_evals = defaultdict(dict)
        for time in t:
            self.known_evals[time] = {}

    @property
    def t(self):
        "Times at which the solution is evaluated"
        return self._t

    @t.setter
    def t(self, value):
        "Updates the solution times"
        self._t = value

    @property
    def y(self):
        "Values of the solution"
        return self._y

    @y.setter
    def y(self, value):
        "Updates the solution values"
        self._y = value

    @property
    def inputs(self):
        "Values of the inputs"
        return self._inputs

    @inputs.setter
    def inputs(self, inputs):
        "Updates the input values"
        self._inputs = {}
        for name, inp in inputs.items():
            if isinstance(inp, numbers.Number):
                inp = inp * np.ones_like(self.t)
            self._inputs[name] = inp

    @property
    def model(self):
        "Model used for solution"
        return self._model

    @model.setter
    def model(self, value):
        "Updates the model"
        assert isinstance(value, pybamm.BaseModel)
        self._model = value

    @property
    def t_event(self):
        "Time at which the event happens"
        return self._t_event

    @t_event.setter
    def t_event(self, value):
        "Updates the event time"
        self._t_event = value

    @property
    def y_event(self):
        "Value of the solution at the time of the event"
        return self._y_event

    @y_event.setter
    def y_event(self, value):
        "Updates the solution at the time of the event"
        self._y_event = value

    @property
    def termination(self):
        "Reason for termination"
        return self._termination

    @termination.setter
    def termination(self, value):
        "Updates the reason for termination"
        self._termination = value

    def append(self, solution):
        """
        Appends solution.t and solution.y onto self.t and self.y.
        Note: this process removes the initial time and state of solution to avoid
        duplicate times and states being stored (self.t[-1] is equal to solution.t[0],
        and self.y[:, -1] is equal to solution.y[:, 0]).

        """
        # Update t, y and inputs
        self.t = np.concatenate((self.t, solution.t[1:]))
        self.y = np.concatenate((self.y, solution.y[:, 1:]), axis=1)
        for name, inp in self.inputs.items():
            solution_inp = solution.inputs[name]
            if isinstance(solution_inp, numbers.Number):
                solution_inp = solution_inp * np.ones_like(solution.t)
            self.inputs[name] = np.concatenate((inp, solution_inp[1:]))
        # Update solution time
        self.solve_time += solution.solve_time

    @property
    def total_time(self):
        return self.set_up_time + self.solve_time

    def update(self, variables):
        """Add ProcessedVariables to the dictionary of variables in the solution"""
        # Convert single entry to list
        if isinstance(variables, str):
            variables = [variables]
        # Process
        for key in variables:
            pybamm.logger.debug("Post-processing {}".format(key))
            var = pybamm.ProcessedVariable(
                self.model.variables[key], self, self.known_evals
            )

            # Update known_evals in order to process any other variables faster
            for t in var.known_evals:
                self.known_evals[t].update(var.known_evals[t])

            # Save variable and data
            self._variables[key] = var
            self.data[key] = var.data

    def __getitem__(self, key):
        """Read a variable from the solution. Variables are created 'just in time', i.e.
        only when they are called.

        Parameters
        ----------
        key : str
            The name of the variable

        Returns
        -------
        :class:`pybamm.ProcessedVariable`
            A variable that can be evaluated at any time or spatial point. The
            underlying data for this variable is available in its attribute ".data"
        """

        try:
            # Try getting item
            # return it if it exists
            return self._variables[key]
        except KeyError:
            # otherwise create it, save it and then return it
            self.update(key)
            return self._variables[key]

    def save(self, filename):
        """Save the whole solution using pickle"""
        # No warning here if len(self.data)==0 as solution can be loaded
        # and used to process new variables
        with open(filename, "wb") as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)

    def save_data(self, filename):
        """Save solution data only (raw arrays) using pickle"""
        if len(self.data) == 0:
            raise ValueError(
                """Solution does not have any data. Add variables by calling
                'solution.update', e.g.
                'solution.update(["Terminal voltage [V]", "Current [A]"])'
                and then save"""
            )
        with open(filename, "wb") as f:
            pickle.dump(self.data, f, pickle.HIGHEST_PROTOCOL)

