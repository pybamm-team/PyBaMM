#
# Summary Variable class
#
from __future__ import annotations
import pybamm
import numpy as np


class SummaryVariables:
    """


    Parameters
    ----------
    solution : :class:`pybamm.Solution`
        The solution object to be used to create the processed variables

    """

    def __init__(
        self, solution, cycle_summary_variables=None, esoh_solver=None, user_inputs=None
    ):
        self.user_inputs = user_inputs or {}
        self.model = solution.all_models[0]
        self.esoh_solver = esoh_solver
        self._esoh_variables = []
        self._variables = {}  # make fuzzy dict?
        self._possible_variables = self.model.summary_variables  # minus esoh variables
        self.cycle_number = None

        # only ever have one SummaryVariables object per solution,
        # even if it has sub-cycles
        if cycle_summary_variables:
            self.first_state = None
            self.last_state = None
            self.cycles = cycle_summary_variables
            self.cycle_number = np.array(range(1, len(self.cycles) + 1))
            self.esoh_solver = self.cycles[0].esoh_solver
            # do these change from cycle to cycle?
            self.user_inputs = self.cycles[0].user_inputs
        else:
            self.first_state = solution.first_state
            self.last_state = solution.last_state
            self.cycles = None

    @property
    def all_variables(self):
        try:
            return self._all_variables
        except AttributeError:

            base_vars = self._possible_variables.copy()
            for var in self._possible_variables:
                var_lowercase = var[0].lower() + var[1:]
                base_vars.append("Change in " + var_lowercase)

            if self._esoh_variables:
                base_vars.extend(self._esoh_variables)
            elif self.esoh_solver is not None:
                _ = self.esoh_solver._get_electrode_soh_sims_full().model
                esoh_variables = _.variables.keys()
                base_vars.extend(esoh_variables)

            self._all_variables = base_vars
            return self._all_variables

    def __getitem__(self, key):
        """Read a variable from the solution. Variables are created 'just in time', i.e.
        only when they are called.

        Parameters
        ----------
        key : str
            The name of the variable

        Returns
        -------
        :class:`pybamm.ProcessedVariable` or :class:`pybamm.ProcessedVariableComputed`
            A variable that can be evaluated at any time or spatial point. The
            underlying data for this variable is available in its attribute ".data"
        """

        base_key = key.removeprefix("Change in ")
        base_key = base_key[0].upper() + base_key[1:]

        # return it if it exists
        if base_key in self._variables or key in self._esoh_variables:
            return self._variables[key]
        elif base_key not in self._possible_variables:
            # check it's not a relevant eSOH variable
            if (
                self.esoh_solver is not None
                and isinstance(self.model, pybamm.lithium_ion.BaseModel)
                and self.model.options.electrode_types["negative"] == "porous"
                and "Negative electrode capacity [A.h]" in self.model.variables
                and "Positive electrode capacity [A.h]" in self.model.variables
            ):
                esoh_sim = self.esoh_solver._get_electrode_soh_sims_full().model
                self._esoh_variables = esoh_sim.variables.keys()
                if key in self._esoh_variables:
                    self.update_esoh()
                    return self._variables[key]
                else:
                    raise KeyError(
                        f"Variable '{key}' is not a summary variable or an eSOH "
                        "variable."
                    )
            else:
                raise KeyError(f"Variable '{key}' is not a summary variable.")
        else:
            # otherwise create it, save it and then return it
            self.update(base_key)
            return self._variables[key]

    def update(self, var):
        var_lowercase = var[0].lower() + var[1:]
        if self.cycles is not None:
            var_cycle = []
            change_var_cycle = []
            for cycle in self.cycles:
                var_cycle.append(cycle[var])
                change_var_cycle.append(cycle["Change in " + var_lowercase])
            self._variables[var] = var_cycle
            self._variables["Change in " + var_lowercase] = change_var_cycle
        else:
            data_first = self.first_state[var].data
            data_last = self.last_state[var].data
            self._variables[var] = data_last[0]
            self._variables["Change in " + var_lowercase] = data_last[0] - data_first[0]

    def update_esoh(self):
        if self.cycles is not None:
            var_cycle = []
            for cycle in self.cycles:
                var_cycle.append(cycle.get_esoh_variables())
            vars = {k: [] for k in var_cycle[0].keys()}
            for cycle in var_cycle:
                for k, v in cycle.items():
                    vars[k].append(v)
            self._variables.update(vars)
        else:
            esoh_vars = self.get_esoh_variables()
            self._variables.update(esoh_vars)

    def get_esoh_variables(self):
        # eSOH variables (full-cell lithium-ion model only, for now)
        Q_n = self.last_state["Negative electrode capacity [A.h]"].data[0]
        Q_p = self.last_state["Positive electrode capacity [A.h]"].data[0]
        Q_Li = self.last_state["Total lithium capacity in particles [A.h]"].data[0]
        all_inputs = {**self.user_inputs, "Q_n": Q_n, "Q_p": Q_p, "Q_Li": Q_Li}
        try:
            esoh_sol = self.esoh_solver.solve(inputs=all_inputs)
        except pybamm.SolverError as error:  # pragma: no cover
            raise pybamm.SolverError(
                "Could not solve for eSOH summary variables"
            ) from error

        return esoh_sol
