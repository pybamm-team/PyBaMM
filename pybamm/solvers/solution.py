#
# Solution class
#
import casadi
import json
import numbers
import numpy as np
import pickle
import pybamm
import pandas as pd
from scipy.io import savemat
from functools import cached_property


class NumpyEncoder(json.JSONEncoder):
    """
    Numpy serialiser helper class that converts numpy arrays to a list.
    Numpy arrays cannot be directly converted to JSON, so the arrays are
    converted to python list objects before encoding.
    """

    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        # won't be called since we only need to convert numpy arrays
        return json.JSONEncoder.default(self, obj)  # pragma: no cover


class Solution:
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

    sensitivities: bool or dict
        True if sensitivities included as the solution of the explicit forwards
        equations.  False if no sensitivities included/wanted. Dict if sensitivities are
        provided as a dict of {parameter: sensitivities} pairs.

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
        sensitivities=False,
        check_solution=True,
    ):
        if not isinstance(all_ts, list):
            all_ts = [all_ts]
        if not isinstance(all_ys, list):
            all_ys = [all_ys]
        if not isinstance(all_models, list):
            all_models = [all_models]
        self._all_ts = all_ts
        self._all_ys = all_ys
        self._all_ys_and_sens = all_ys
        self._all_models = all_models

        # Set up inputs
        if not isinstance(all_inputs, list):
            all_inputs_copy = dict(all_inputs)
            for key, value in all_inputs_copy.items():
                if isinstance(value, numbers.Number):
                    all_inputs_copy[key] = np.array([value])
            self.all_inputs = [all_inputs_copy]
        else:
            self.all_inputs = all_inputs

        self.sensitivities = sensitivities

        # Check no ys are too large
        if check_solution:
            self.check_ys_are_not_too_large()

        # Events
        self._t_event = t_event
        self._y_event = y_event
        self._termination = termination
        self.closest_event_idx = None

        # Initialize times
        self.set_up_time = None
        self.solve_time = None
        self.integration_time = None

        # initialize empty variables and data
        self._variables = pybamm.FuzzyDict()
        self.data = pybamm.FuzzyDict()

        # Add self as sub-solution for compatibility with ProcessedVariable
        self._sub_solutions = [self]

        # initialize empty cycles
        self._cycles = []

        # Initialize empty summary variables
        self._summary_variables = None

        # Initialise initial start time
        self.initial_start_time = None

        # Solution now uses CasADi
        pybamm.citations.register("Andersson2019")

    def extract_explicit_sensitivities(self):
        # if we got here, we haven't set y yet
        self.set_y()

        # extract sensitivities from full y solution
        self._y, self._sensitivities = self._extract_explicit_sensitivities(
            self.all_models[0], self.y, self.t, self.all_inputs[0]
        )

        # make sure we remove all sensitivities from all_ys
        for index, (model, ys, ts, inputs) in enumerate(
            zip(self.all_models, self.all_ys, self.all_ts, self.all_inputs)
        ):
            self._all_ys[index], _ = self._extract_explicit_sensitivities(
                model, ys, ts, inputs
            )

    def _extract_explicit_sensitivities(self, model, y, t_eval, inputs):
        """
        given a model and a solution y, extracts the sensitivities

        Parameters
        --------
        model : :class:`pybamm.BaseModel`
            A model that has been already setup by this base solver
        y: ndarray
            The solution of the full explicit sensitivity equations
        t_eval: ndarray
            The evaluation times
        inputs: dict
            parameter inputs

        Returns
        -------
        y: ndarray
            The solution of the ode/dae in model
        sensitivities: dict of (string: ndarray)
            A dictionary of parameter names, and the corresponding solution of
            the sensitivity equations
        """

        n_states = model.len_rhs_and_alg
        n_rhs = model.len_rhs
        n_alg = model.len_alg
        # Get the point where the algebraic equations start
        if model.len_rhs != 0:
            n_p = model.len_rhs_sens // model.len_rhs
        else:
            n_p = model.len_alg_sens // model.len_alg
        len_rhs_and_sens = model.len_rhs + model.len_rhs_sens

        n_t = len(t_eval)
        # y gets the part of the solution vector that correspond to the
        # actual ODE/DAE solution

        # save sensitivities as a dictionary
        # first save the whole sensitivity matrix
        # reshape using Fortran order to get the right array:
        #   t0_x0_p0, t0_x0_p1, ..., t0_x0_pn
        #   t0_x1_p0, t0_x1_p1, ..., t0_x1_pn
        #   ...
        #   t0_xn_p0, t0_xn_p1, ..., t0_xn_pn
        #   t1_x0_p0, t1_x0_p1, ..., t1_x0_pn
        #   t1_x1_p0, t1_x1_p1, ..., t1_x1_pn
        #   ...
        #   t1_xn_p0, t1_xn_p1, ..., t1_xn_pn
        #   ...
        #   tn_x0_p0, tn_x0_p1, ..., tn_x0_pn
        #   tn_x1_p0, tn_x1_p1, ..., tn_x1_pn
        #   ...
        #   tn_xn_p0, tn_xn_p1, ..., tn_xn_pn
        # 1, Extract rhs and alg sensitivities and reshape into 3D matrices
        # with shape (n_p, n_states, n_t)
        if isinstance(y, casadi.DM):
            y_full = y.full()
        else:
            y_full = y
        ode_sens = y_full[n_rhs:len_rhs_and_sens, :].reshape(n_p, n_rhs, n_t)
        alg_sens = y_full[len_rhs_and_sens + n_alg :, :].reshape(n_p, n_alg, n_t)
        # 2. Concatenate into a single 3D matrix with shape (n_p, n_states, n_t)
        # i.e. along first axis
        full_sens_matrix = np.concatenate([ode_sens, alg_sens], axis=1)
        # Transpose and reshape into a (n_states * n_t, n_p) matrix
        full_sens_matrix = full_sens_matrix.transpose(2, 1, 0).reshape(
            n_t * n_states, n_p
        )

        # Save the full sensitivity matrix
        sensitivity = {"all": full_sens_matrix}

        # also save the sensitivity wrt each parameter (read the columns of the
        # sensitivity matrix)
        start = 0
        for name in model.calculate_sensitivities:
            inp = inputs[name]
            input_size = inp.shape[0]
            end = start + input_size
            sensitivity[name] = full_sens_matrix[:, start:end]
            start = end

        y_dae = np.vstack(
            [
                y[: model.len_rhs, :],
                y[len_rhs_and_sens : len_rhs_and_sens + model.len_alg, :],
            ]
        )
        return y_dae, sensitivity

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

            # if y is evaluated before sensitivities then need to extract them
            if isinstance(self._sensitivities, bool) and self._sensitivities:
                self.extract_explicit_sensitivities()

            return self._y

    @property
    def sensitivities(self):
        """Values of the sensitivities. Returns a dict of param_name: np_array"""
        if isinstance(self._sensitivities, bool):
            if self._sensitivities:
                self.extract_explicit_sensitivities()
            else:
                self._sensitivities = {}
        return self._sensitivities

    @sensitivities.setter
    def sensitivities(self, value):
        """Updates the sensitivity"""
        # sensitivities must be a dict or bool
        if not isinstance(value, (bool, dict)):
            raise TypeError("sensitivities arg needs to be a bool or dict")
        self._sensitivities = value

    def set_y(self):
        try:
            if isinstance(self.all_ys[0], (casadi.DM, casadi.MX)):
                self._y = casadi.horzcat(*self.all_ys)
            else:
                self._y = np.hstack(self.all_ys)
        except ValueError as error:
            raise pybamm.SolverError(
                "The solution is made up from different models, so `y` cannot be "
                "computed explicitly."
            ) from error

    def check_ys_are_not_too_large(self):
        # Only check last one so that it doesn't take too long
        # We only care about the cases where y is growing too large without any
        # restraint, so if y gets large in the middle then comes back down that is ok
        y, model = self.all_ys[-1], self.all_models[-1]
        y = y[:, -1]
        if np.any(y > pybamm.settings.max_y_value):
            for var in [*model.rhs.keys(), *model.algebraic.keys()]:
                var = model.variables[var.name]
                # find the statevector corresponding to this variable
                statevector = None
                for node in var.pre_order():
                    if isinstance(node, pybamm.StateVector):
                        statevector = node

                # there will always be a statevector, but just in case
                if statevector is None:  # pragma: no cover
                    raise RuntimeError(
                        f"Cannot find statevector corresponding to variable {var.name}"
                    )
                y_var = y[statevector.y_slices[0]]
                if np.any(y_var > pybamm.settings.max_y_value):
                    pybamm.logger.error(
                        f"Solution for '{var}' exceeds the maximum allowed value "
                        f"of `{pybamm.settings.max_y_value}. This could be due to "
                        "incorrect scaling, model formulation, or "
                        "parameter values. The maximum allowed value is set by "
                        "'pybammm.settings.max_y_value'."
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

    @cached_property
    def all_inputs_casadi(self):
        return [casadi.vertcat(*inp.values()) for inp in self.all_inputs]

    @property
    def t_event(self):
        """Time at which the event happens"""
        return self._t_event

    @property
    def y_event(self):
        """Value of the solution at the time of the event"""
        return self._y_event

    @property
    def termination(self):
        """Reason for termination"""
        return self._termination

    @termination.setter
    def termination(self, value):
        """Updates the reason for termination"""
        self._termination = value

    @cached_property
    def first_state(self):
        """
        A Solution object that only contains the first state. This is faster to evaluate
        than the full solution when only the first state is needed (e.g. to initialize
        a model with the solution)
        """
        new_sol = Solution(
            self.all_ts[0][:1],
            self.all_ys[0][:, :1],
            self.all_models[:1],
            self.all_inputs[:1],
            None,
            None,
            "final time",
        )
        new_sol._all_inputs_casadi = self.all_inputs_casadi[:1]
        new_sol._sub_solutions = self.sub_solutions[:1]

        new_sol.solve_time = 0
        new_sol.integration_time = 0
        new_sol.set_up_time = 0

        return new_sol

    @cached_property
    def last_state(self):
        """
        A Solution object that only contains the final state. This is faster to evaluate
        than the full solution when only the final state is needed (e.g. to initialize
        a model with the solution)
        """
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
        new_sol._sub_solutions = self.sub_solutions[-1:]

        new_sol.solve_time = 0
        new_sol.integration_time = 0
        new_sol.set_up_time = 0

        return new_sol

    @property
    def total_time(self):
        return self.set_up_time + self.solve_time

    @property
    def cycles(self):
        return self._cycles

    @cycles.setter
    def cycles(self, cycles):
        self._cycles = cycles

    @property
    def summary_variables(self):
        return self._summary_variables

    @property
    def initial_start_time(self):
        return self._initial_start_time

    @initial_start_time.setter
    def initial_start_time(self, value):
        """Updates the initial start time of the experiment"""
        self._initial_start_time = value

    def set_summary_variables(self, all_summary_variables):
        summary_variables = {var: [] for var in all_summary_variables[0]}
        for sum_vars in all_summary_variables:
            for name, value in sum_vars.items():
                summary_variables[name].append(value)

        summary_variables["Cycle number"] = range(1, len(all_summary_variables) + 1)
        self.all_summary_variables = all_summary_variables
        self._summary_variables = pybamm.FuzzyDict(
            {name: np.array(value) for name, value in summary_variables.items()}
        )

    def update(self, variables):
        """Add ProcessedVariables to the dictionary of variables in the solution"""
        # make sure that sensitivities are extracted if required
        if isinstance(self._sensitivities, bool) and self._sensitivities:
            self.extract_explicit_sensitivities()

        # Convert single entry to list
        if isinstance(variables, str):
            variables = [variables]
        # Process
        for key in variables:
            cumtrapz_ic = None
            pybamm.logger.debug(f"Post-processing {key}")
            vars_pybamm = [model.variables_and_events[key] for model in self.all_models]

            # Iterate through all models, some may be in the list several times and
            # therefore only get set up once
            vars_casadi = []
            for i, (model, ys, inputs, var_pybamm) in enumerate(
                zip(self.all_models, self.all_ys, self.all_inputs, vars_pybamm)
            ):
                if ys.size == 0 and var_pybamm.has_symbol_of_classes(
                    pybamm.expression_tree.state_vector.StateVector
                ):
                    raise KeyError(
                        f"Cannot process variable '{key}' as it was not part of the "
                        "solve. Please re-run the solve with `output_variables` set to "
                        "include this variable."
                    )
                elif isinstance(var_pybamm, pybamm.ExplicitTimeIntegral):
                    cumtrapz_ic = var_pybamm.initial_condition
                    cumtrapz_ic = cumtrapz_ic.evaluate()
                    var_pybamm = var_pybamm.child
                    var_casadi = self.process_casadi_var(
                        var_pybamm,
                        inputs,
                        ys.shape,
                    )
                    model._variables_casadi[key] = var_casadi
                    vars_pybamm[i] = var_pybamm
                elif key in model._variables_casadi:
                    var_casadi = model._variables_casadi[key]
                else:
                    var_casadi = self.process_casadi_var(
                        var_pybamm,
                        inputs,
                        ys.shape,
                    )
                    model._variables_casadi[key] = var_casadi
                vars_casadi.append(var_casadi)
            var = pybamm.ProcessedVariable(
                vars_pybamm, vars_casadi, self, cumtrapz_ic=cumtrapz_ic
            )

            # Save variable and data
            self._variables[key] = var
            self.data[key] = var.data

    def process_casadi_var(self, var_pybamm, inputs, ys_shape):
        t_MX = casadi.MX.sym("t")
        y_MX = casadi.MX.sym("y", ys_shape[0])
        inputs_MX_dict = {
            key: casadi.MX.sym("input", value.shape[0]) for key, value in inputs.items()
        }
        inputs_MX = casadi.vertcat(*[p for p in inputs_MX_dict.values()])
        var_sym = var_pybamm.to_casadi(t_MX, y_MX, inputs=inputs_MX_dict)
        var_casadi = casadi.Function("variable", [t_MX, y_MX, inputs_MX], [var_sym])
        return var_casadi

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

    def save(self, filename):
        """Save the whole solution using pickle"""
        # No warning here if len(self.data)==0 as solution can be loaded
        # and used to process new variables

        with open(filename, "wb") as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)

    def get_data_dict(self, variables=None, short_names=None, cycles_and_steps=True):
        """
        Construct a (standard python) dictionary of the solution data containing the
        variables in `variables`. If `variables` is None then all variables are
        returned. Any variable names in short_names are replaced with the corresponding
        short name.

        If the solution has cycles, then the cycle numbers and step numbers are also
        returned in the dictionary.

        Parameters
        ----------
        variables : list, optional
            List of variables to return. If None, returns all variables in solution.data
        short_names : dict, optional
            Dictionary of shortened names to use when saving.
        cycles_and_steps : bool, optional
            Whether to include the cycle numbers and step numbers in the dictionary

        Returns
        -------
        dict
            A dictionary of the solution data
        """
        if variables is None:
            # variables not explicitly provided -> save all variables that have been
            # computed
            data_long_names = self.data
        else:
            if isinstance(variables, str):
                variables = [variables]
            # otherwise, save only the variables specified
            data_long_names = {}
            for name in variables:
                data_long_names[name] = self[name].data
        if len(data_long_names) == 0:
            raise ValueError(
                """
                Solution does not have any data. Please provide a list of variables
                to save.
                """
            )

        # Use any short names if provided
        data_short_names = {}
        short_names = short_names or {}
        for name, var in data_long_names.items():
            name = short_names.get(name, name)  # return name if no short name
            data_short_names[name] = var

        # Save cycle number and step number if the solution has them
        if cycles_and_steps and len(self.cycles) > 0:
            data_short_names["Cycle"] = np.array([])
            data_short_names["Step"] = np.array([])
            for i, cycle in enumerate(self.cycles):
                data_short_names["Cycle"] = np.concatenate(
                    [data_short_names["Cycle"], i * np.ones_like(cycle.t)]
                )
                for j, step in enumerate(cycle.steps):
                    data_short_names["Step"] = np.concatenate(
                        [data_short_names["Step"], j * np.ones_like(step.t)]
                    )

        return data_short_names

    def save_data(
        self, filename=None, variables=None, to_format="pickle", short_names=None
    ):
        """
        Save solution data only (raw arrays)

        Parameters
        ----------
        filename : str, optional
            The name of the file to save data to. If None, then a str is returned
        variables : list, optional
            List of variables to save. If None, saves all of the variables that have
            been created so far
        to_format : str, optional
            The format to save to. Options are:

            - 'pickle' (default): creates a pickle file with the data dictionary
            - 'matlab': creates a .mat file, for loading in matlab
            - 'csv': creates a csv file (0D variables only)
            - 'json': creates a json file
        short_names : dict, optional
            Dictionary of shortened names to use when saving. This may be necessary when
            saving to MATLAB, since no spaces or special characters are allowed in
            MATLAB variable names. Note that not all the variables need to be given
            a short name.

        Returns
        -------
        data : str, optional
            str if 'csv' or 'json' is chosen and filename is None, otherwise None
        """
        data = self.get_data_dict(variables=variables, short_names=short_names)

        if to_format == "pickle":
            if filename is None:
                raise ValueError("pickle format must be written to a file")
            with open(filename, "wb") as f:
                pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
        elif to_format == "matlab":
            if filename is None:
                raise ValueError("matlab format must be written to a file")
            # Check all the variable names only contain a-z, A-Z or _ or numbers
            for name in data.keys():
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
                            f"Invalid character '{s}' found in '{name}'. "
                            + "MATLAB variable names must only contain a-z, A-Z, _, "
                            "or 0-9 (except the first position). "
                            "Use the 'short_names' argument to pass an alternative "
                            "variable name, e.g. \n\n"
                            "\tsolution.save_data(filename, "
                            "['Electrolyte concentration'], to_format='matlab, "
                            "short_names={'Electrolyte concentration': 'c_e'})"
                        )
            savemat(filename, data)
        elif to_format == "csv":
            for name, var in data.items():
                if var.ndim >= 2:
                    raise ValueError(
                        f"only 0D variables can be saved to csv, but '{name}' is {var.ndim - 1}D"
                    )
            df = pd.DataFrame(data)
            return df.to_csv(filename, index=False)
        elif to_format == "json":
            if filename is None:
                return json.dumps(data, cls=NumpyEncoder)
            else:
                with open(filename, "w") as outfile:
                    json.dump(data, outfile, cls=NumpyEncoder)
        else:
            raise ValueError(f"format '{to_format}' not recognised")

    @property
    def sub_solutions(self):
        """List of sub solutions that have been
        concatenated to form the full solution"""

        return self._sub_solutions

    def __add__(self, other):
        """Adds two solutions together, e.g. when stepping"""
        if other is None or isinstance(other, EmptySolution):
            return self.copy()
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
            new_sol = self.copy()
            # Update termination using the latter solution
            new_sol._termination = other.termination
            new_sol._t_event = other._t_event
            new_sol._y_event = other._y_event
            return new_sol

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
            other.t_event,
            other.y_event,
            other.termination,
            bool(self.sensitivities),
        )

        new_sol.closest_event_idx = other.closest_event_idx
        new_sol._all_inputs_casadi = self.all_inputs_casadi + other.all_inputs_casadi

        # Set solution time
        new_sol.solve_time = self.solve_time + other.solve_time
        new_sol.integration_time = self.integration_time + other.integration_time

        # Set sub_solutions
        new_sol._sub_solutions = self.sub_solutions + other.sub_solutions

        return new_sol

    def __radd__(self, other):
        return self.__add__(other)

    def copy(self):
        new_sol = self.__class__(
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
        new_sol.closest_event_idx = self.closest_event_idx

        new_sol.solve_time = self.solve_time
        new_sol.integration_time = self.integration_time
        new_sol.set_up_time = self.set_up_time

        return new_sol

    def plot_voltage_components(
        self,
        ax=None,
        show_legend=True,
        split_by_electrode=False,
        show_plot=True,
        **kwargs_fill,
    ):
        """
        Generate a plot showing the component overpotentials that make up the voltage

        Parameters
        ----------
        ax : matplotlib Axis, optional
            The axis on which to put the plot. If None, a new figure and axis is created.
        show_legend : bool, optional
            Whether to display the legend. Default is True.
        split_by_electrode : bool, optional
            Whether to show the overpotentials for the negative and positive electrodes
            separately. Default is False.
        show_plot : bool, optional
            Whether to show the plots. Default is True. Set to False if you want to
            only display the plot after plt.show() has been called.
        kwargs_fill
            Keyword arguments, passed to ax.fill_between.

        """
        # Use 'self' here as the solution object
        return pybamm.plot_voltage_components(
            self,
            ax=ax,
            show_legend=show_legend,
            split_by_electrode=split_by_electrode,
            show_plot=show_plot,
            **kwargs_fill,
        )


class EmptySolution:
    def __init__(self, termination=None, t=None):
        self.termination = termination
        if t is None:
            t = np.array([0])
        elif isinstance(t, numbers.Number):
            t = np.array([t])

        self.t = t

    def __add__(self, other):
        if isinstance(other, (EmptySolution, Solution)):
            return other.copy()

    def __radd__(self, other):
        if other is None:
            return self.copy()

    def copy(self):
        return EmptySolution(termination=self.termination, t=self.t)


def make_cycle_solution(
    step_solutions, esoh_solver=None, save_this_cycle=True, inputs=None
):
    """
    Function to create a Solution for an entire cycle, and associated summary variables

    Parameters
    ----------
    step_solutions : list of :class:`Solution`
        Step solutions that form the entire cycle
    esoh_solver : :class:`pybamm.lithium_ion.ElectrodeSOHSolver`
        Solver to calculate electrode SOH (eSOH) variables. If `None` (default)
        then only summary variables that do not require the eSOH calculation
        are calculated. See :footcite:t:`Mohtat2019` for more details on eSOH variables.
    save_this_cycle : bool, optional
        Whether to save the entire cycle variables or just the summary variables.
        Default True

    Returns
    -------
    cycle_solution : :class:`pybamm.Solution` or None
        The Solution object for this cycle, or None (if save_this_cycle is False)
    cycle_summary_variables : dict
        Dictionary of summary variables for this cycle

    """
    sum_sols = step_solutions[0].copy()
    for step_solution in step_solutions[1:]:
        sum_sols = sum_sols + step_solution

    cycle_solution = Solution(
        sum_sols.all_ts,
        sum_sols.all_ys,
        sum_sols.all_models,
        sum_sols.all_inputs,
        sum_sols.t_event,
        sum_sols.y_event,
        sum_sols.termination,
    )
    cycle_solution._all_inputs_casadi = sum_sols.all_inputs_casadi
    cycle_solution._sub_solutions = sum_sols.sub_solutions

    cycle_solution.solve_time = sum_sols.solve_time
    cycle_solution.integration_time = sum_sols.integration_time
    cycle_solution.set_up_time = sum_sols.set_up_time

    cycle_solution.steps = step_solutions

    cycle_summary_variables = _get_cycle_summary_variables(
        cycle_solution, esoh_solver, user_inputs=inputs
    )

    cycle_first_state = cycle_solution.first_state

    if save_this_cycle:
        cycle_solution.cycle_summary_variables = cycle_summary_variables
    else:
        cycle_solution = None

    return cycle_solution, cycle_summary_variables, cycle_first_state


def _get_cycle_summary_variables(cycle_solution, esoh_solver, user_inputs=None):
    user_inputs = user_inputs or {}
    model = cycle_solution.all_models[0]
    cycle_summary_variables = pybamm.FuzzyDict({})

    # Summary variables
    summary_variables = model.summary_variables
    first_state = cycle_solution.first_state
    last_state = cycle_solution.last_state
    for var in summary_variables:
        data_first = first_state[var].data
        data_last = last_state[var].data
        cycle_summary_variables[var] = data_last[0]
        var_lowercase = var[0].lower() + var[1:]
        cycle_summary_variables["Change in " + var_lowercase] = (
            data_last[0] - data_first[0]
        )

    # eSOH variables (full-cell lithium-ion model only, for now)
    if (
        esoh_solver is not None
        and isinstance(model, pybamm.lithium_ion.BaseModel)
        and model.options.electrode_types["negative"] == "porous"
        and "Negative electrode capacity [A.h]" in model.variables
        and "Positive electrode capacity [A.h]" in model.variables
    ):
        Q_n = last_state["Negative electrode capacity [A.h]"].data[0]
        Q_p = last_state["Positive electrode capacity [A.h]"].data[0]
        Q_Li = last_state["Total lithium capacity in particles [A.h]"].data[0]
        all_inputs = {**user_inputs, "Q_n": Q_n, "Q_p": Q_p, "Q_Li": Q_Li}
        try:
            esoh_sol = esoh_solver.solve(inputs=all_inputs)
        except pybamm.SolverError as error:  # pragma: no cover
            raise pybamm.SolverError(
                "Could not solve for summary variables, run "
                "`sim.solve(calc_esoh=False)` to skip this step"
            ) from error

        cycle_summary_variables.update(esoh_sol)

    return cycle_summary_variables
