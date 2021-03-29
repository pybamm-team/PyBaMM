#
# Solver class using sundials with the KLU sparse linear solver
#
import casadi
import pybamm
import numpy as np
import scipy.sparse as sparse

import importlib

idaklu_spec = importlib.util.find_spec("pybamm.solvers.idaklu")
if idaklu_spec is not None:
    idaklu = importlib.util.module_from_spec(idaklu_spec)
    idaklu_spec.loader.exec_module(idaklu)


def have_idaklu():
    return idaklu_spec is not None


class IDAKLUSolver(pybamm.BaseSolver):
    """Solve a discretised model, using sundials with the KLU sparse linear solver.

     Parameters
    ----------
    rtol : float, optional
        The relative tolerance for the solver (default is 1e-6).
    atol : float, optional
        The absolute tolerance for the solver (default is 1e-6).
    root_method : str or pybamm algebraic solver class, optional
        The method to use to find initial conditions (for DAE solvers).
        If a solver class, must be an algebraic solver class.
        If "casadi",
        the solver uses casadi's Newton rootfinding algorithm to find initial
        conditions. Otherwise, the solver uses 'scipy.optimize.root' with method
        specified by 'root_method' (e.g. "lm", "hybr", ...)
    root_tol : float, optional
        The tolerance for the initial-condition solver (default is 1e-6).
    extrap_tol : float, optional
        The tolerance to assert whether extrapolation occurs or not (default is 0).
    """

    def __init__(
        self,
        rtol=1e-6,
        atol=1e-6,
        root_method="casadi",
        root_tol=1e-6,
        extrap_tol=0,
        max_steps="deprecated",
    ):

        if idaklu_spec is None:
            raise ImportError("KLU is not installed")

        super().__init__(
            "ida", rtol, atol, root_method, root_tol, extrap_tol, max_steps
        )
        self.name = "IDA KLU solver"

        pybamm.citations.register("Hindmarsh2000")
        pybamm.citations.register("Hindmarsh2005")

    def set_atol_by_variable(self, variables_with_tols, model):
        """
        A method to set the absolute tolerances in the solver by state variable.
        This method attaches a vector of tolerance to the model. (i.e. model.atol)

        Parameters
        ----------
        variables_with_tols : dict
            A dictionary with keys that are strings indicating the variable you
            wish to set the tolerance of and values that are the tolerances.

        model : :class:`pybamm.BaseModel`
            The model that is going to be solved.
        """

        size = model.concatenated_initial_conditions.size
        atol = self._check_atol_type(self._atol, size)
        for var, tol in variables_with_tols.items():
            variable = model.variables[var]
            if isinstance(variable, pybamm.StateVector):
                atol = self.set_state_vec_tol(atol, variable, tol)
            else:
                raise pybamm.SolverError("Can only set tolerances for state variables")

        model.atol = atol

    def set_state_vec_tol(self, atol, state_vec, tol):
        """
        A method to set the tolerances in the atol vector of a specific
        state variable. This method modifies self._atol

        Parameters
        ----------
        state_vec : :class:`pybamm.StateVector`
            The state vector to apply to the tolerance to
        tol: float
            The tolerance value
        """
        slices = state_vec.y_slices[0]
        atol[slices] = tol
        return atol

    def _check_atol_type(self, atol, size):
        """
        This method checks that the atol vector is of the right shape and
        type.

        Parameters
        ----------
        atol: double or np.array or list
            Absolute tolerances. If this is a vector then each entry corresponds to
            the absolute tolerance of one entry in the state vector.
        size: int
            The length of the atol vector
        """

        if isinstance(atol, float):
            atol = atol * np.ones(size)
        elif isinstance(atol, list):
            atol = np.array(atol)
        elif isinstance(atol, np.ndarray):
            pass
        else:
            raise pybamm.SolverError(
                "Absolute tolerances must be a numpy array, float, or list"
            )

        if atol.size != size:
            raise pybamm.SolverError(
                """Absolute tolerances must be either a scalar or a numpy arrray
                of the same shape at y0"""
            )

        return atol

    def _integrate(self, model, t_eval, inputs_dict=None):
        """
        Solve a DAE model defined by residuals with initial conditions y0.

        Parameters
        ----------
        model : :class:`pybamm.BaseModel`
            The model whose solution to calculate.
        t_eval : numeric type
            The times at which to compute the solution
        inputs_dict : dict, optional
            Any external variables or input parameters to pass to the model when solving
        """
        if model.rhs_eval.form == "casadi":
            # stack inputs
            inputs = casadi.vertcat(*[x for x in inputs_dict.values()])
        else:
            inputs = inputs_dict

        if model.jacobian_eval is None:
            raise pybamm.SolverError("KLU requires the Jacobian to be provided")

        try:
            atol = model.atol
        except AttributeError:
            atol = self._atol

        y0 = model.y0
        if isinstance(y0, casadi.DM):
            y0 = y0.full().flatten()

        rtol = self._rtol
        atol = self._check_atol_type(atol, y0.size)

        mass_matrix = model.mass_matrix.entries

        if model.jacobian_eval:
            jac_y0_t0 = model.jacobian_eval(t_eval[0], y0, inputs)
            if sparse.issparse(jac_y0_t0):

                def jacfn(t, y, cj):
                    j = model.jacobian_eval(t, y, inputs) - cj * mass_matrix
                    return j

            else:

                def jacfn(t, y, cj):
                    jac_eval = model.jacobian_eval(t, y, inputs) - cj * mass_matrix
                    return sparse.csr_matrix(jac_eval)

        class SundialsJacobian:
            def __init__(self):
                self.J = None

                random = np.random.random(size=y0.size)
                J = jacfn(10, random, 20)
                self.nnz = J.nnz  # hoping nnz remains constant...

            def jac_res(self, t, y, cj):
                # must be of form j_res = (dr/dy) - (cj) (dr/dy')
                # cj is just the input parameter
                # see p68 of the ida_guide.pdf for more details
                self.J = jacfn(t, y, cj)

            def get_jac_data(self):
                return self.J.data

            def get_jac_row_vals(self):
                return self.J.indices

            def get_jac_col_ptrs(self):
                return self.J.indptr

        # solver works with ydot0 set to zero
        ydot0 = np.zeros_like(y0)

        jac_class = SundialsJacobian()

        num_of_events = len(model.terminate_events_eval)
        use_jac = 1

        def rootfn(t, y):
            return_root = np.ones((num_of_events,))
            return_root[:] = [
                event(t, y, inputs) for event in model.terminate_events_eval
            ]

            return return_root

        # get ids of rhs and algebraic variables
        rhs_ids = np.ones(model.rhs_eval(0, y0, inputs).shape)
        alg_ids = np.zeros(len(y0) - len(rhs_ids))
        ids = np.concatenate((rhs_ids, alg_ids))

        # solve
        timer = pybamm.Timer()
        sol = idaklu.solve(
            t_eval,
            y0,
            ydot0,
            lambda t, y, ydot: model.residuals_eval(t, y, ydot, inputs),
            jac_class.jac_res,
            jac_class.get_jac_data,
            jac_class.get_jac_row_vals,
            jac_class.get_jac_col_ptrs,
            jac_class.nnz,
            rootfn,
            num_of_events,
            use_jac,
            ids,
            atol,
            rtol,
        )
        integration_time = timer.time()

        t = sol.t
        number_of_timesteps = t.size
        number_of_states = y0.size
        y_out = sol.y.reshape((number_of_timesteps, number_of_states))

        # return solution, we need to tranpose y to match scipy's interface
        if sol.flag in [0, 2]:
            # 0 = solved for all t_eval
            if sol.flag == 0:
                termination = "final time"
            # 2 = found root(s)
            elif sol.flag == 2:
                termination = "event"

            sol = pybamm.Solution(
                sol.t,
                np.transpose(y_out),
                model,
                inputs_dict,
                t[-1],
                np.transpose(y_out[-1])[:, np.newaxis],
                termination,
            )
            sol.integration_time = integration_time
            return sol
        else:
            raise pybamm.SolverError(sol.message)
