#
# Solution class
#
import casadi
import copy
import numbers
import numpy as np
import pickle
import pybamm
import pandas as pd
from collections import defaultdict
from scipy.io import savemat


class _BaseSolution(object):
    """
    (Semi-private) class containing the solution of, and various attributes associated
    with, a PyBaMM model. This class is automatically created by the `Solution` class,
    and should never be called from outside the `Solution` class.

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
    copy_this : :class:`pybamm.Solution`, optional
        A solution to copy, if provided. Default is None.

    """

    def __init__(
        self, t, y, t_event=None, y_event=None, termination="final time", copy_this=None
    ):
        self._t = t
        if isinstance(y, casadi.DM):
            y = y.full()
        self._y = y
        self._t_event = t_event
        self._y_event = y_event
        self._termination = termination
        if copy_this is None:
            # initialize empty inputs and model, to be populated later
            self._inputs = pybamm.FuzzyDict()
            self._model = pybamm.BaseModel()
            self.set_up_time = None
            self.solve_time = None
            self.has_symbolic_inputs = False
        else:
            self._inputs = copy.copy(copy_this.inputs)
            self._model = copy_this.model
            self.set_up_time = copy_this.set_up_time
            self.solve_time = copy_this.solve_time
            self.has_symbolic_inputs = copy_this.has_symbolic_inputs

        # initiaize empty variables and data
        self._variables = pybamm.FuzzyDict()
        self.data = pybamm.FuzzyDict()

        # initialize empty known evals
        self._known_evals = defaultdict(dict)
        for time in t:
            self._known_evals[time] = {}

    @property
    def t(self):
        "Times at which the solution is evaluated"
        return self._t

    @property
    def y(self):
        "Values of the solution"
        return self._y

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
    def inputs(self):
        "Values of the inputs"
        return self._inputs

    @inputs.setter
    def inputs(self, inputs):
        "Updates the input values"
        # If there are symbolic inputs, just store them as given
        if any(isinstance(v, casadi.MX) for v in inputs.values()):
            self.has_symbolic_inputs = True
            self._inputs = inputs
        # Otherwise, make them the same size as the time vector
        else:
            self.has_symbolic_inputs = False
            self._inputs = {}
            for name, inp in inputs.items():
                # Convert number to vector of the right shape
                if isinstance(inp, numbers.Number):
                    inp = inp * np.ones((1, len(self.t)))
                # Tile a vector
                else:
                    inp = np.tile(inp, len(self.t))
                self._inputs[name] = inp

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
            # If there are symbolic inputs then we need to make a
            # ProcessedSymbolicVariable
            if self.has_symbolic_inputs is True:
                var = pybamm.ProcessedSymbolicVariable(self.model.variables[key], self)

            # Otherwise a standard ProcessedVariable is ok
            else:
                var = pybamm.ProcessedVariable(
                    self.model.variables[key], self, self._known_evals
                )

                # Update known_evals in order to process any other variables faster
                for t in var.known_evals:
                    self._known_evals[t].update(var.known_evals[t])

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

        # return it if it exists
        if key in self._variables:
            return self._variables[key]
        else:
            # otherwise create it, save it and then return it
            self.update(key)
            return self._variables[key]

    def save(self, filename):
        """Save the whole solution using pickle"""
        # No warning here if len(self.data)==0 as solution can be loaded
        # and used to process new variables
        with open(filename, "wb") as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)

    def save_data(self, filename, variables=None, to_format="pickle"):
        """
        Save solution data only (raw arrays)

        Parameters
        ----------
        filename : str
            The name of the file to save data to
        variables : list, optional
            List of variables to save. If None, saves all of the variables that have
            been created so far
        to_format : str, optional
            The format to save to. Options are:

            - 'pickle' (default): creates a pickle file with the data dictionary
            - 'matlab': creates a .mat file, for loading in matlab
            - 'csv': creates a csv file (1D variables only)

        """
        if variables is None:
            # variables not explicitly provided -> save all variables that have been
            # computed
            data = self.data
        else:
            # otherwise, save only the variables specified
            data = {}
            for name in variables:
                data[name] = self[name].data
        if len(data) == 0:
            raise ValueError(
                """
                Solution does not have any data. Please provide a list of variables
                to save.
                """
            )
        if to_format == "pickle":
            with open(filename, "wb") as f:
                pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
        elif to_format == "matlab":
            savemat(filename, data)
        elif to_format == "csv":
            for name, var in data.items():
                if var.ndim >= 2:
                    raise ValueError(
                        "only 0D variables can be saved to csv, but '{}' is {}D".format(
                            name, var.ndim - 1
                        )
                    )
            df = pd.DataFrame(data)
            df.to_csv(filename, index=False)


class Solution(_BaseSolution):
    """
    Class extending the base solution, with additional functionality for concatenating
    different solutions together

    **Extends**: :class:`_BaseSolution`

    """

    def __init__(self, t, y, t_event=None, y_event=None, termination="final time"):
        super().__init__(t, y, t_event, y_event, termination)
        self.base_solution_class = _BaseSolution

    @property
    def sub_solutions(self):
        "List of sub solutions that have been concatenated to form the full solution"
        try:
            return self._sub_solutions
        except AttributeError:
            raise AttributeError(
                "sub solutions are only created once other solutions have been appended"
            )

    def __add__(self, other):
        "See :meth:`Solution.append`"
        self.append(other, create_sub_solutions=True)
        return self

    def append(self, solution, start_index=1, create_sub_solutions=False):
        """
        Appends solution.t and solution.y onto self.t and self.y.

        Note: by default this process removes the initial time and state of solution to
        avoid duplicate times and states being stored (self.t[-1] is equal to
        solution.t[0], and self.y[:, -1] is equal to solution.y[:, 0]). Set the optional
        argument ``start_index`` to override this behavior
        """
        # Create sub-solutions if necessary
        # sub-solutions are 'BaseSolution' objects, which have slightly reduced
        # functionality compared to normal solutions (can't append other solutions)
        if create_sub_solutions and not hasattr(self, "_sub_solutions"):
            self._sub_solutions = [
                self.base_solution_class(
                    self.t,
                    self.y,
                    self.t_event,
                    self.y_event,
                    self.termination,
                    copy_this=self,
                )
            ]

        # (Create and) update sub-solutions
        # Create a list of sub-solutions, which are simpler BaseSolution classes

        # Update t, y and inputs
        self._t = np.concatenate((self._t, solution.t[start_index:]))
        self._y = np.concatenate((self._y, solution.y[:, start_index:]), axis=1)
        for name, inp in self.inputs.items():
            solution_inp = solution.inputs[name]
            self.inputs[name] = np.c_[inp, solution_inp[:, start_index:]]
        # Update solution time
        self.solve_time += solution.solve_time
        # Update termination
        self._termination = solution.termination
        self._t_event = solution._t_event
        self._y_event = solution._y_event

        # Update known_evals
        for t, evals in solution._known_evals.items():
            self._known_evals[t].update(evals)
        # Recompute existing variables
        for var in self._variables.keys():
            self.update(var)

        # Append sub_solutions
        if create_sub_solutions:
            self._sub_solutions.append(
                self.base_solution_class(
                    solution.t,
                    solution.y,
                    solution.t_event,
                    solution.y_event,
                    solution.termination,
                    copy_this=solution,
                )
            )
