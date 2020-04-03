#
# Processed Variable class
#
import casadi
import numbers
import numpy as np
import pybamm


class ProcessedCasadiVariable(object):
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
        # Checks
        if not isinstance(solution, pybamm.CasadiSolution):
            raise TypeError(
                "solution must be a 'CasadiSolution' but is {}".format(solution)
            )

        # Convert variable to casadi
        t_MX = casadi.MX.sym("t")
        y_MX = casadi.MX.sym("y", solution.y.shape[0])
        inputs_MX = casadi.vertcat(*[p for p in solution.inputs.values()])
        var = base_variable.to_casadi(t_MX, y_MX, inputs=solution.inputs)

        self.base_variable = casadi.Function("variable", [t_MX, y_MX, inputs_MX], [var])
        self.t_sol = solution.t
        self.u_sol = solution.y
        self.mesh = base_variable.mesh
        self.input_keys = solution.inputs.keys()
        self.inputs = inputs_MX
        self.domain = base_variable.domain

        self.base_eval = self.base_variable(
            solution.t[0], solution.y[:, 0], self.inputs,
        )

        if (
            isinstance(self.base_eval, numbers.Number)
            or len(self.base_eval.shape) == 0
            or self.base_eval.shape[0] == 1
        ):
            self.initialise_0D()
        else:
            n = self.mesh[0].npts
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
        self.casadi_entries_fn = casadi.Function("variable", [inputs_MX], [entries_MX])

        # Don't compute jacobian if the entries are a DM (not symbolic)
        if isinstance(entries_MX, casadi.DM):
            self.casadi_sens_fn = None
        # Do compute jacobian if the entries are symbolic (functions of input)
        else:
            sens_MX = casadi.jacobian(entries_MX, inputs_MX)
            self.casadi_sens_fn = casadi.Function("variable", [inputs_MX], [sens_MX])

    def initialise_0D(self):
        # Evaluate the base_variable index-by-index
        for idx in range(len(self.t_sol)):
            t = self.t_sol[idx]
            u = self.u_sol[:, idx]
            next_entries = self.base_variable(t, u, self.inputs)
            if idx == 0:
                entries = next_entries
            else:
                entries = casadi.horzcat(entries, next_entries)

        self.entries = entries
        self.dimensions = 0

    def initialise_1D(self):
        len_space = self.base_eval.shape[0]
        entries = np.empty((len_space, len(self.t_sol)))

        # Evaluate the base_variable index-by-index
        for idx in range(len(self.t_sol)):
            t = self.t_sol[idx]
            u = self.u_sol[:, idx]
            next_entries = self.base_variable(t, u, self.inputs).flatten()
            if idx == 0:
                entries = next_entries
            else:
                entries = casadi.horzcat(entries, next_entries)

        # Get node values
        nodes = self.mesh[0].nodes

        # assign attributes for reference (either x_sol or r_sol)
        self.entries = entries
        self.dimensions = 1
        if self.domain[0] in ["negative particle", "positive particle"]:
            self.first_dimension = "r"
            self.r_sol = nodes
        elif self.domain[0] in [
            "negative electrode",
            "separator",
            "positive electrode",
        ]:
            self.first_dimension = "x"
            self.x_sol = nodes
        elif self.domain == ["current collector"]:
            self.first_dimension = "z"
            self.z_sol = nodes
        else:
            self.first_dimension = "x"
            self.x_sol = nodes

        self.first_dim_pts = nodes

    def value(self, inputs=None, check_inputs=True):
        if inputs is None:
            return self.casadi_entries_fn(casadi.DM())
        else:
            if check_inputs:
                inputs = self.check_and_transform(inputs)
            return self.casadi_entries_fn(inputs)

    def sensitivity(self, inputs=None, check_inputs=True):
        if self.casadi_sens_fn is None:
            raise ValueError(
                "Variable is not symbolic, so sensitivities are not defined"
            )
        if check_inputs:
            inputs = self.check_and_transform(inputs)
        return self.casadi_sens_fn(inputs)

    def value_and_sensitivity(self, inputs=None):
        inputs = self.check_and_transform(inputs)
        return (
            self.value(inputs, check_inputs=False),
            self.sensitivity(inputs, check_inputs=False),
        )

    def check_and_transform(self, inputs):
        # Convert dict to casadi vector
        if isinstance(inputs, dict):
            # Check keys are consistent
            if inputs.keys() != self.input_keys:
                raise ValueError(
                    "Inconsistent input keys: expected {}, actual {}".format(
                        inputs.keys(), self.input_keys
                    )
                )
            inputs = casadi.vertcat(*[p for p in inputs.values()])

        return inputs

