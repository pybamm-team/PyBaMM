#
# Processed Variable class
#
import casadi
import numbers
import numpy as np


class ProcessedSymbolicVariable(object):
    """
    An object that can be evaluated at arbitrary (scalars or vectors) t and x, and
    returns the (interpolated) value of the base variable at that t and x.

    Parameters
    ----------
    base_variable : :class:`pybamm.Symbol`
        A base variable with a method `evaluate(t,y)` that returns the value of that
        variable. Note that this can be any kind of node in the expression tree, not
        just a :class:`pybamm.Variable`.
        When evaluated, returns an array of size (m,n)
    solution : :class:`pybamm.Solution`
        The solution object to be used to create the processed variables
    """

    def __init__(self, base_variable, solution):
        # Convert variable to casadi
        t_MX = casadi.MX.sym("t")
        y_MX = casadi.MX.sym("y", solution.all_ys[0].shape[0])
        # Make all inputs symbolic first for converting to casadi
        all_inputs_as_MX_dict = {}
        symbolic_inputs_dict = {}
        for key, value in solution.all_inputs[0].items():
            if not isinstance(value, casadi.MX):
                all_inputs_as_MX_dict[key] = casadi.MX.sym("input")
            else:
                all_inputs_as_MX_dict[key] = value
                # Only add symbolic inputs to the "symbolic_inputs" dict
                symbolic_inputs_dict[key] = value

        all_inputs_as_MX = casadi.vertcat(*[p for p in all_inputs_as_MX_dict.values()])
        # The symbolic_inputs dictionary will be used for sensitivity
        symbolic_inputs = casadi.vertcat(*[p for p in symbolic_inputs_dict.values()])
        var = base_variable.to_casadi(t_MX, y_MX, inputs=all_inputs_as_MX_dict)

        self.base_variable = casadi.Function(
            "variable", [t_MX, y_MX, all_inputs_as_MX], [var]
        )
        # Store some attributes
        self.t_pts = solution.t
        self.all_ts = solution.all_ts
        self.all_ys = solution.all_ys
        self.mesh = base_variable.mesh
        self.all_inputs_casadi = solution.all_inputs_casadi

        self.symbolic_inputs_dict = symbolic_inputs_dict
        self.symbolic_inputs_total_shape = symbolic_inputs.shape[0]
        self.domain = base_variable.domain

        self.base_eval = self.base_variable(
            self.all_ts[0][0], self.all_ys[0][:, 0], self.all_inputs_casadi[0]
        )

        if (
            isinstance(self.base_eval, numbers.Number)
            or len(self.base_eval.shape) == 0
            or self.base_eval.shape[0] == 1
        ):
            self.initialise_0D()
        else:
            n = self.mesh.npts
            base_shape = self.base_eval.shape[0]
            # Try shape that could make the variable a 1D variable
            if base_shape == n:
                self.initialise_1D()
            else:
                # Raise error for 2D variable
                raise NotImplementedError(
                    "Shape not recognized for {} ".format(base_variable)
                    + "(note processing of 2D and 3D variables is not yet "
                    + "implemented)"
                )

        # Make entries a function and compute jacobian
        entries_MX = self.entries
        self.casadi_entries_fn = casadi.Function(
            "variable", [symbolic_inputs], [entries_MX]
        )

        # Don't compute jacobian if the entries are a DM (not symbolic)
        if isinstance(entries_MX, casadi.DM):
            self.casadi_sens_fn = None
        # Do compute jacobian if the entries are symbolic (functions of input)
        else:
            sens_MX = casadi.jacobian(entries_MX, symbolic_inputs)
            self.casadi_sens_fn = casadi.Function(
                "variable", [symbolic_inputs], [sens_MX]
            )

    def initialise_0D(self):
        """Create a 0D variable"""
        # Evaluate the base_variable index-by-index
        idx = 0
        for ts, ys, inputs in zip(self.all_ts, self.all_ys, self.all_inputs_casadi):
            for inner_idx, t in enumerate(ts):
                t = ts[inner_idx]
                y = ys[:, inner_idx]
                next_entries = self.base_variable(t, y, inputs)
                if idx == 0:
                    entries = next_entries
                else:
                    entries = casadi.horzcat(entries, next_entries)
                idx += 1

        self.entries = entries
        self.dimensions = 0

    def initialise_1D(self):
        """Create a 1D variable"""
        len_space = self.base_eval.shape[0]
        entries = np.empty((len_space, len(self.t_pts)))

        # Evaluate the base_variable index-by-index
        idx = 0
        for ts, ys, inputs in zip(self.all_ts, self.all_ys, self.all_inputs_casadi):
            for inner_idx, t in enumerate(ts):
                t = ts[inner_idx]
                y = ys[:, inner_idx]
                next_entries = self.base_variable(t, y, inputs)
                if idx == 0:
                    entries = next_entries
                else:
                    entries = casadi.vertcat(entries, next_entries)
                idx += 1

        self.entries = entries

    def value(self, inputs=None, check_inputs=True):
        """
        Returns the value of the variable at the specified input values

        Parameters
        ----------
        inputs : dict
            The inputs at which to evaluate the variable.

        Returns
        -------
        casadi.DM
            A casadi matrix of size (n_x * n_t, 1), where n_x is the number of spatial
            discretisation points for the variable, and n_t is the length of the time
            vector
        """
        if inputs is None:
            return self.casadi_entries_fn(casadi.DM())
        else:
            if check_inputs:
                inputs = self._check_and_transform(inputs)
            return self.casadi_entries_fn(inputs)

    def sensitivity(self, inputs=None, check_inputs=True):
        """
        Returns the sensitivity of the variable to the symbolic inputs at the specified
        input values

        Parameters
        ----------
        inputs : dict
            The inputs at which to evaluate the variable.

        Returns
        -------
        casadi.DM
            A casadi matrix of size (n_x * n_t, n_p), where n_x is the number of spatial
            discretisation points for the variable, n_t is the length of the time
            vector, and n_p is the number of input parameters
        """
        if self.casadi_sens_fn is None:
            raise ValueError(
                "Variable is not symbolic, so sensitivities are not defined"
            )
        if check_inputs:
            inputs = self._check_and_transform(inputs)
        return self.casadi_sens_fn(inputs)

    def value_and_sensitivity(self, inputs=None):
        """
        Returns the value of the variable and its sensitivity to the symbolic inputs at
        the specified input values

        Parameters
        ----------
        inputs : dict
            The inputs at which to evaluate the variable.
        """
        inputs = self._check_and_transform(inputs)
        # Pass check_inputs=False to avoid re-checking inputs
        return (
            self.value(inputs, check_inputs=False),
            self.sensitivity(inputs, check_inputs=False),
        )

    def _check_and_transform(self, inputs_dict):
        """Check dictionary has the right inputs, and convert to a vector"""
        # Convert dict to casadi vector
        if not isinstance(inputs_dict, dict):
            raise TypeError("inputs should be 'dict' but are {}".format(inputs_dict))
        # Sort input dictionary keys according to the symbolic inputs dictionary
        # For practical number of input parameters this should be extremely fast and
        # so is ok to do at each step
        try:
            inputs_dict_sorted = {
                k: inputs_dict[k] for k in self.symbolic_inputs_dict.keys()
            }
        except KeyError as e:
            raise KeyError("Inconsistent input keys. '{}' not found".format(e.args[0]))
        inputs = casadi.vertcat(*[p for p in inputs_dict_sorted.values()])
        if inputs.shape[0] != self.symbolic_inputs_total_shape:
            # Find the variable which caused the error, for a clearer error message
            for key, inp in inputs_dict_sorted.items():
                if inp.shape[0] != self.symbolic_inputs_dict[key].shape[0]:
                    raise ValueError(
                        "Wrong shape for input '{}': expected {}, actual {}".format(
                            key, self.symbolic_inputs_dict[key].shape[0], inp.shape[0]
                        )
                    )

        return inputs

    @property
    def data(self):
        """Same as entries, but different name"""
        return self.entries
