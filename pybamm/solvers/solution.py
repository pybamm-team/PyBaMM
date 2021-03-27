#
# Solution class
#
import casadi
import numbers
import numpy as np
import pickle
import pybamm
import pandas as pd
from scipy.io import savemat


class Solution(object):
    """
    Class containing the solution of, and various attributes associated with, a PyBaMM
    model.

    Parameters
    ----------
    all_ts : :class:`numpy.array`, size (n,) (or list of these)
        A one-dimensional array containing the times at which the solution is evaluated.
        A list of times can be provided instead to initialize a solution with
        sub-solutions.
    all_ys : :class:`numpy.array`, size (m, n) (or list of these)
        A two-dimensional array containing the values of the solution. y[i, :] is the
        vector of solutions at time t[i].
        A list of ys can be provided instead to initialize a solution with
        sub-solutions.
    all_models : :class:`pybamm.BaseModel`
        The model that was used to calculate the solution.
        A list of models can be provided instead to initialize a solution with
        sub-solutions that have been calculated using those models.
    all_inputs : dict (or list of these)
        The inputs that were used to calculate the solution
        A list of inputs can be provided instead to initialize a solution with
        sub-solutions.
    t_event : :class:`numpy.array`, size (1,)
        A zero-dimensional array containing the time at which the event happens.
    y_event : :class:`numpy.array`, size (m,)
        A one-dimensional array containing the value of the solution at the time when
        the event happens.
    termination : str
        String to indicate why the solution terminated

    """

    def __init__(
        self,
        all_ts,
        all_ys,
        all_models,
        all_inputs,
        t_event=None,
        y_event=None,
        termination="final time",
    ):
        if not isinstance(all_ts, list):
            all_ts = [all_ts]
        if not isinstance(all_ys, list):
            all_ys = [all_ys]
        if not isinstance(all_models, list):
            all_models = [all_models]
        self._all_ts = all_ts
        self._all_ys = all_ys
        self._all_models = all_models

        self._t_event = t_event
        self._y_event = y_event
        self._termination = termination

        # Set up inputs
        if not isinstance(all_inputs, list):
            for key, value in all_inputs.items():
                if isinstance(value, numbers.Number):
                    all_inputs[key] = np.array([value])
            all_inputs = [all_inputs]
        self.all_inputs = all_inputs
        self.has_symbolic_inputs = any(
            isinstance(v, casadi.MX) for v in all_inputs[0].values()
        )

        # Copy the timescale_eval and lengthscale_evals if they exist
        if hasattr(all_models[0], "timescale_eval"):
            self.timescale_eval = all_models[0].timescale_eval
        else:
            self.timescale_eval = all_models[0].timescale.evaluate()

        if hasattr(all_models[0], "length_scales_eval"):
            self.length_scales_eval = all_models[0].length_scales_eval
        else:
            self.length_scales_eval = {
                domain: scale.evaluate()
                for domain, scale in all_models[0].length_scales.items()
            }

        self.set_up_time = None
        self.solve_time = None
        self.integration_time = None

        # initiaize empty variables and data
        self._variables = pybamm.FuzzyDict()
        self.data = pybamm.FuzzyDict()

        # Add self as sub-solution for compatibility with ProcessedVariable
        self._sub_solutions = [self]

        # Solution now uses CasADi
        pybamm.citations.register("Andersson2019")

    @property
    def t(self):
        """Times at which the solution is evaluated"""
        try:
            return self._t
        except AttributeError:
            self.set_t()
            return self._t

    def set_t(self):
        self._t = np.concatenate(self.all_ts)
        if any(np.diff(self._t) <= 0):
            raise ValueError("Solution time vector must be strictly increasing")

    @property
    def y(self):
        """Values of the solution"""
        try:
            return self._y
        except AttributeError:
            self.set_y()
            return self._y

    def set_y(self):
        try:
            if isinstance(self.all_ys[0], (casadi.DM, casadi.MX)):
                self._y = casadi.horzcat(*self.all_ys)
            else:
                self._y = np.hstack(self.all_ys)
        except ValueError:
            raise pybamm.SolverError(
                "The solution is made up from different models, so `y` cannot be "
                "computed explicitly."
            )

    @property
    def all_ts(self):
        return self._all_ts

    @property
    def all_ys(self):
        return self._all_ys

    @property
    def all_models(self):
        """Model(s) used for solution"""
        return self._all_models

    @property
    def all_inputs_casadi(self):
        try:
            return self._all_inputs_casadi
        except AttributeError:
            self._all_inputs_casadi = [
                casadi.vertcat(*inp.values()) for inp in self.all_inputs
            ]
            return self._all_inputs_casadi

    @property
    def t_event(self):
        """Time at which the event happens"""
        return self._t_event

    @t_event.setter
    def t_event(self, value):
        """Updates the event time"""
        self._t_event = value

    @property
    def y_event(self):
        """Value of the solution at the time of the event"""
        return self._y_event

    @y_event.setter
    def y_event(self, value):
        """Updates the solution at the time of the event"""
        self._y_event = value

    @property
    def termination(self):
        """Reason for termination"""
        return self._termination

    @termination.setter
    def termination(self, value):
        """Updates the reason for termination"""
        self._termination = value

    @property
    def last_state(self):
        """
        A Solution object that only contains the final state. This is faster to evaluate
        than the full solution when only the final state is needed (e.g. to initialize
        a model with the solution)
        """
        try:
            return self._last_state
        except AttributeError:
            new_sol = Solution(
                self.all_ts[-1][-1:],
                self.all_ys[-1][:, -1:],
                self.all_models[-1:],
                self.all_inputs[-1:],
                self.t_event,
                self.y_event,
                self.termination,
            )
            new_sol._all_inputs_casadi = self.all_inputs_casadi[-1:]
            new_sol._sub_solutions = self.sub_solutions

            new_sol.solve_time = 0
            new_sol.integration_time = 0
            new_sol.set_up_time = 0

            self._last_state = new_sol
            return self._last_state

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
                var = pybamm.ProcessedSymbolicVariable(
                    self.all_models[0].variables[key], self
                )

            # Otherwise a standard ProcessedVariable is ok
            else:
                vars_pybamm = [model.variables[key] for model in self.all_models]

                # Iterate through all models, some may be in the list several times and
                # therefore only get set up once
                vars_casadi = []
                for model, ys, inputs, var_pybamm in zip(
                    self.all_models, self.all_ys, self.all_inputs, vars_pybamm
                ):
                    if key in model._variables_casadi:
                        var_casadi = model._variables_casadi[key]
                    else:
                        t_MX = casadi.MX.sym("t")
                        y_MX = casadi.MX.sym("y", ys.shape[0])
                        symbolic_inputs_dict = {
                            key: casadi.MX.sym("input", value.shape[0])
                            for key, value in inputs.items()
                        }
                        symbolic_inputs = casadi.vertcat(
                            *[p for p in symbolic_inputs_dict.values()]
                        )

                        # Convert variable to casadi
                        # Make all inputs symbolic first for converting to casadi
                        var_sym = var_pybamm.to_casadi(
                            t_MX, y_MX, inputs=symbolic_inputs_dict
                        )

                        var_casadi = casadi.Function(
                            "variable", [t_MX, y_MX, symbolic_inputs], [var_sym]
                        )
                        model._variables_casadi[key] = var_casadi
                    vars_casadi.append(var_casadi)

                var = pybamm.ProcessedVariable(vars_pybamm, vars_casadi, self)

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

    def plot(self, output_variables=None, **kwargs):
        """
        A method to quickly plot the outputs of the solution. Creates a
        :class:`pybamm.QuickPlot` object (with keyword arguments 'kwargs') and
        then calls :meth:`pybamm.QuickPlot.dynamic_plot`.

        Parameters
        ----------
        output_variables: list, optional
            A list of the variables to plot.
        **kwargs
            Additional keyword arguments passed to
            :meth:`pybamm.QuickPlot.dynamic_plot`.
            For a list of all possible keyword arguments see :class:`pybamm.QuickPlot`.
        """
        return pybamm.dynamic_plot(self, output_variables=output_variables, **kwargs)

    def clear_casadi_attributes(self):
        """Remove casadi objects for pickling, will be computed again automatically"""
        # t_MX = None
        # y_MX = None
        # symbolic_inputs = None
        # symbolic_inputs_dict = None
        pass

    def save(self, filename):
        """Save the whole solution using pickle"""
        # No warning here if len(self.data)==0 as solution can be loaded
        # and used to process new variables

        self.clear_casadi_attributes()
        # Pickle
        with open(filename, "wb") as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)

    def save_data(self, filename, variables=None, to_format="pickle", short_names=None):
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
            - 'csv': creates a csv file (0D variables only)
        short_names : dict, optional
            Dictionary of shortened names to use when saving. This may be necessary when
            saving to MATLAB, since no spaces or special characters are allowed in
            MATLAB variable names. Note that not all the variables need to be given
            a short name.

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

        # Use any short names if provided
        data_short_names = {}
        short_names = short_names or {}
        for name, var in data.items():
            # change to short name if it exists
            if name in short_names:
                data_short_names[short_names[name]] = var
            else:
                data_short_names[name] = var

        if to_format == "pickle":
            with open(filename, "wb") as f:
                pickle.dump(data_short_names, f, pickle.HIGHEST_PROTOCOL)
        elif to_format == "matlab":
            # Check all the variable names only contain a-z, A-Z or _ or numbers
            for name in data_short_names.keys():
                # Check the string only contains the following ASCII:
                # a-z (97-122)
                # A-Z (65-90)
                # _ (95)
                # 0-9 (48-57) but not in the first position
                for i, s in enumerate(name):
                    if not (
                        97 <= ord(s) <= 122
                        or 65 <= ord(s) <= 90
                        or ord(s) == 95
                        or (i > 0 and 48 <= ord(s) <= 57)
                    ):
                        raise ValueError(
                            "Invalid character '{}' found in '{}'. ".format(s, name)
                            + "MATLAB variable names must only contain a-z, A-Z, _, "
                            "or 0-9 (except the first position). "
                            "Use the 'short_names' argument to pass an alternative "
                            "variable name, e.g. \n\n"
                            "\tsolution.save_data(filename, "
                            "['Electrolyte concentration'], to_format='matlab, "
                            "short_names={'Electrolyte concentration': 'c_e'})"
                        )
            savemat(filename, data_short_names)
        elif to_format == "csv":
            for name, var in data_short_names.items():
                if var.ndim >= 2:
                    raise ValueError(
                        "only 0D variables can be saved to csv, but '{}' is {}D".format(
                            name, var.ndim - 1
                        )
                    )
            df = pd.DataFrame(data_short_names)
            df.to_csv(filename, index=False)
        else:
            raise ValueError("format '{}' not recognised".format(to_format))

    @property
    def sub_solutions(self):
        """List of sub solutions that have been
        concatenated to form the full solution"""

        return self._sub_solutions

    def __add__(self, other):
        """ Adds two solutions together, e.g. when stepping """
        if not isinstance(other, Solution):
            raise pybamm.SolverError(
                "Only a Solution or None can be added to a Solution"
            )
        # Special case: new solution only has one timestep and it is already in the
        # existing solution. In this case, return a copy of the existing solution
        if (
            len(other.all_ts) == 1
            and len(other.all_ts[0]) == 1
            and other.all_ts[0][0] == self.all_ts[-1][-1]
        ):
            return self.copy()

        # Update list of sub-solutions
        if other.all_ts[0][0] == self.all_ts[-1][-1]:
            # Skip first time step if it is repeated
            all_ts = self.all_ts + [other.all_ts[0][1:]] + other.all_ts[1:]
            all_ys = self.all_ys + [other.all_ys[0][:, 1:]] + other.all_ys[1:]
        else:
            all_ts = self.all_ts + other.all_ts
            all_ys = self.all_ys + other.all_ys

        new_sol = Solution(
            all_ts,
            all_ys,
            self.all_models + other.all_models,
            self.all_inputs + other.all_inputs,
            self.t_event,
            self.y_event,
            self.termination,
        )

        new_sol._all_inputs_casadi = self.all_inputs_casadi + other.all_inputs_casadi

        # Set solution time
        new_sol.solve_time = self.solve_time + other.solve_time
        new_sol.integration_time = self.integration_time + other.integration_time

        # Update termination using the latter solution
        new_sol._termination = other.termination
        new_sol._t_event = other._t_event
        new_sol._y_event = other._y_event

        # Set sub_solutions
        new_sol._sub_solutions = self.sub_solutions + other.sub_solutions

        return new_sol

    def __radd__(self, other):
        """
        Function to deal with the case `None + Solution` (returns `Solution`)
        """
        if other is None:
            return self.copy()
        else:
            raise pybamm.SolverError(
                "Only a Solution or None can be added to a Solution"
            )

    def copy(self):
        new_sol = Solution(
            self.all_ts,
            self.all_ys,
            self.all_models,
            self.all_inputs,
            self.t_event,
            self.y_event,
            self.termination,
        )
        new_sol._all_inputs_casadi = self.all_inputs_casadi
        new_sol._sub_solutions = self.sub_solutions

        new_sol.solve_time = self.solve_time
        new_sol.integration_time = self.integration_time
        new_sol.set_up_time = self.set_up_time

        return new_sol
