#
# Processed Variable class
#
import numbers
import numpy as np
import pybamm
import scipy.interpolate as interp


def post_process_variables(variables, t_sol, y_sol, mesh=None, interp_kind="linear"):
    """
    Post-process all variables in a model

    Parameters
    ----------
    variables : dict
        Dictionary of variables
    t_sol : array_like, size (m,)
        The time vector returned by the solver
    y_sol : array_like, size (m, k)
        The solution vector returned by the solver. Can include solution values that
        other than those that get read by base_variable.evaluate() (i.e. k>=n)
    mesh : :class:`pybamm.Mesh`
        The mesh used to solve, used here to calculate the reference x values for
        interpolation
    interp_kind : str
        The method to use for interpolation

    Returns
    -------
    dict
        Dictionary of processed variables
    """
    processed_variables = {}
    known_evals = {t: {} for t in t_sol}
    for var, eqn in variables.items():
        pybamm.logger.debug("Post-processing {}".format(var))
        try:
            processed_variables[var] = ProcessedVariable(
                eqn, t_sol, y_sol, mesh, interp_kind, known_evals
            )
        except:
            processed_variables[var] = ProcessedVariable(
                eqn, t_sol, y_sol, mesh, interp_kind, known_evals
            )

        for t in known_evals:
            known_evals[t].update(processed_variables[var].known_evals[t])
    return processed_variables


class ProcessedVariable(object):
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
    t_sol : array_like, size (m,)
        The time vector returned by the solver
    y_sol : array_like, size (m, k)
        The solution vector returned by the solver. Can include solution values that
        other than those that get read by base_variable.evaluate() (i.e. k>=n)
    mesh : :class:`pybamm.Mesh`
        The mesh used to solve, used here to calculate the reference x values for
        interpolation
    interp_kind : str
        The method to use for interpolation
    """

    def __init__(
        self,
        base_variable,
        t_sol,
        y_sol,
        mesh=None,
        interp_kind="linear",
        known_evals=None,
    ):
        self.base_variable = base_variable
        self.t_sol = t_sol
        self.y_sol = y_sol
        self.mesh = mesh
        self.interp_kind = interp_kind
        self.domain = base_variable.domain
        self.known_evals = known_evals

        if self.known_evals:
            self.base_eval, self.known_evals[t_sol[0]] = base_variable.evaluate(
                t_sol[0], y_sol[:, 0], self.known_evals[t_sol[0]]
            )
        else:
            self.base_eval = base_variable.evaluate(t_sol[0], y_sol[:, 0])

        if (
            isinstance(self.base_eval, numbers.Number)
            or len(self.base_eval.shape) == 0
            or self.base_eval.shape[0] == 1
        ):
            self.initialise_1D()
        else:
            n = self.mesh.combine_submeshes(*self.domain)[0].npts
            base_shape = self.base_eval.shape[0]
            if base_shape in [n, n + 1]:
                self.initialise_2D()
            else:
                self.initialise_3D()

        # Remove base_variable attribute to allow pickling
        del self.base_variable

    def initialise_1D(self):
        # initialise empty array of the correct size
        entries = np.empty(len(self.t_sol))
        # Evaluate the base_variable index-by-index
        for idx in range(len(self.t_sol)):
            t = self.t_sol[idx]
            if self.known_evals:
                entries[idx], self.known_evals[t] = self.base_variable.evaluate(
                    t, self.y_sol[:, idx], self.known_evals[t]
                )
            else:
                entries[idx] = self.base_variable.evaluate(t, self.y_sol[:, idx])

        # No discretisation provided, or variable has no domain (function of t only)
        self._interpolation_function = interp.interp1d(
            self.t_sol,
            entries,
            kind=self.interp_kind,
            fill_value=np.nan,
            bounds_error=False,
        )

        self.entries = entries
        self.t_x_r_sol = (self.t_sol, None, None)
        self.dimensions = 1

    def initialise_2D(self):
        len_space = self.base_eval.shape[0]
        entries = np.empty((len_space, len(self.t_sol)))

        # Evaluate the base_variable index-by-index
        for idx in range(len(self.t_sol)):
            t = self.t_sol[idx]
            y = self.y_sol[:, idx]
            if self.known_evals:
                eval_and_known_evals = self.base_variable.evaluate(
                    t, y, self.known_evals[t]
                )
                entries[:, idx] = eval_and_known_evals[0][:, 0]
                self.known_evals[t] = eval_and_known_evals[1]
            else:
                entries[:, idx] = self.base_variable.evaluate(t, y)[:, 0]

        # Process the discretisation to get x values
        nodes = self.mesh.combine_submeshes(*self.domain)[0].nodes
        edges = self.mesh.combine_submeshes(*self.domain)[0].edges
        if entries.shape[0] == len(nodes):
            space = nodes
        elif entries.shape[0] == len(edges):
            space = edges

        # add points outside domain for extrapolation to boundaries
        extrap_space_left = np.array([2 * space[0] - space[1]])
        extrap_space_right = np.array([2 * space[-1] - space[-2]])
        space = np.concatenate([extrap_space_left, space, extrap_space_right])
        extrap_entries_left = 2 * entries[0] - entries[1]
        extrap_entries_right = 2 * entries[-1] - entries[-2]
        entries = np.vstack([extrap_entries_left, entries, extrap_entries_right])

        # assign attributes for reference (either x_sol or r_sol)
        self.entries = entries
        self.dimensions = 2
        if any("particle" in dom for dom in self.domain):
            self.scale = "micro"
            self.r_sol = space
            self.t_x_r_sol = (self.t_sol, None, self.r_sol)
        else:
            self.scale = "macro"
            self.x_sol = space
            self.t_x_r_sol = (self.t_sol, self.x_sol, None)

        # set up interpolation
        # note that the order of 't' and 'space' is the reverse of what you'd expect

        self._interpolation_function = interp.interp2d(
            self.t_sol, space, entries, kind=self.interp_kind, fill_value=np.nan
        )

    def initialise_3D(self):
        if self.base_variable.has_symbol_of_class(pybamm.Outer):
            # bit of a hack for now
            len_x = self.mesh.combine_submeshes(*self.domain)[0].npts
        else:
            len_x = len(self.mesh.combine_submeshes(*self.domain))
        len_r = self.base_eval.shape[0] // len_x
        entries = np.empty((len_x, len_r, len(self.t_sol)))

        # Evaluate the base_variable index-by-index
        for idx in range(len(self.t_sol)):
            t = self.t_sol[idx]
            y = self.y_sol[:, idx]
            if self.known_evals:
                eval_and_known_evals = self.base_variable.evaluate(
                    t, y, self.known_evals[t]
                )

                # another hack sorry
                if self.base_variable.has_symbol_of_class(pybamm.Outer):
                    temporary = np.reshape(eval_and_known_evals[0], [len_r, len_x])
                    entries[:, :, idx] = np.transpose(temporary)
                else:
                    entries[:, :, idx] = np.reshape(
                        eval_and_known_evals[0], [len_x, len_r]
                    )

                self.known_evals[t] = eval_and_known_evals[1]
            else:
                entries[:, :, idx] = np.reshape(
                    self.base_variable.evaluate(t, y), [len_x, len_r]
                )
        # Process the discretisation to get x values
        if self.domain == [
            "negative electrode"
        ] and self.base_variable.has_symbol_of_class(pybamm.Outer):
            nodes = self.mesh.combine_submeshes(*["negative particle"])[0].nodes
            edges = self.mesh.combine_submeshes(*["negative particle"])[0].edges
        elif self.domain == [
            "positive electrode"
        ] and self.base_variable.has_symbol_of_class(pybamm.Outer):
            nodes = self.mesh.combine_submeshes(*["positive particle"])[0].nodes
            edges = self.mesh.combine_submeshes(*["positive particle"])[0].edges
        else:
            nodes = self.mesh.combine_submeshes(*self.domain)[0].nodes
            edges = self.mesh.combine_submeshes(*self.domain)[0].edges

        if entries.shape[1] == len(nodes):
            r_sol = nodes
        elif entries.shape[1] == len(edges):
            r_sol = edges
        else:
            raise ValueError("3D variable shape does not match domain shape")

        # Get x values
        if self.domain == ["negative particle"] or self.domain == [
            "negative electrode"
        ]:
            x_sol = self.mesh["negative electrode"][0].nodes
        elif self.domain == ["positive particle"] or self.domain == [
            "positive electrode"
        ]:
            x_sol = self.mesh["positive electrode"][0].nodes

        # assign attributes for reference
        self.entries = entries
        self.dimensions = 3
        self.x_sol = x_sol
        self.r_sol = r_sol
        self.t_x_r_sol = (self.t_sol, x_sol, r_sol)

        # set up interpolation
        self._interpolation_function = interp.RegularGridInterpolator(
            (x_sol, r_sol, self.t_sol),
            entries,
            method=self.interp_kind,
            fill_value=np.nan,
        )

    def __call__(self, t, x=None, r=None):
        "Evaluate the variable at arbitrary t (and x and/or r), using interpolation"
        if self.dimensions == 1:
            return self._interpolation_function(t)
        elif self.dimensions == 2:
            if self.scale == "micro":
                return self._interpolation_function(t, r)
            else:
                return self._interpolation_function(t, x)
        elif self.dimensions == 3:
            if isinstance(x, np.ndarray):
                if isinstance(r, np.ndarray) and isinstance(t, np.ndarray):
                    x = x[:, np.newaxis, np.newaxis]
                    r = r[:, np.newaxis]
                elif isinstance(r, np.ndarray) or isinstance(t, np.ndarray):
                    x = x[:, np.newaxis]
            else:
                if isinstance(r, np.ndarray) and isinstance(t, np.ndarray):
                    r = r[:, np.newaxis]

            return self._interpolation_function((x, r, t))
