#
# Solver class using sundials with the KLU sparse linear solver
#
import pybamm
import numpy as np
import scipy.sparse as sparse

import importlib

idaklu_spec = importlib.util.find_spec("idaklu")
if idaklu_spec is not None:
    idaklu = importlib.util.module_from_spec(idaklu_spec)
    idaklu_spec.loader.exec_module(idaklu)


def have_idaklu():
    return idaklu_spec is None


class IDAKLUSolver(pybamm.DaeSolver):
    """Solve a discretised model, using sundials with the KLU sparse linear solver.

     Parameters
    ----------
    rtol : float, optional
        The relative tolerance for the solver (default is 1e-6).
    atol : float, optional
        The absolute tolerance for the solver (default is 1e-6).
    root_method : str, optional
        The method to use to find initial conditions (default is "lm")
    root_tol : float, optional
        The tolerance for the initial-condition solver (default is 1e-8).
    max_steps: int, optional
        The maximum number of steps the solver will take before terminating
        (default is 1000).
    """

    def __init__(
        self, rtol=1e-6, atol=1e-6, root_method="lm", root_tol=1e-6, max_steps=1000
    ):

        if idaklu_spec is None:
            raise ImportError("KLU is not installed")

        super().__init__("ida", rtol, atol, root_method, root_tol, max_steps)
        self.name = "IDA KLU solver"

    @property
    def atol(self):
        return self._atol

    @atol.setter
    def atol(self, variables_with_tols, model):
        """
        A method to set the absolute tolerances in the solver by state variable.
        This method modifies self._atol.

        Parameters
        ----------
        variables_with_tols : dict
            A dictionary with keys that are strings indicating the variable you
            wish to set the tolerance of and values that are the tolerances.

        model : :class:`pybamm.BaseModel`
            The model that is going to be solved.
        """

        size = model.concatenated_initial_conditions.size
        self._check_atol_type(size)
        for var, tol in variables_with_tols.items():
            variable = model.variables[var]
            if isinstance(variable, pybamm.StateVector):
                self.set_state_vec_tol(variable, tol)
            elif isinstance(variable, pybamm.Concatenation):
                for child in variable.children:
                    if isinstance(child, pybamm.StateVector):
                        self.set_state_vec_tol(child, tol)
                    else:
                        raise pybamm.SolverError(
                            """Can only set tolerances for state variables
                            or concatenations of state variables"""
                        )
            else:
                raise pybamm.SolverError(
                    """Can only set tolerances for state variables or
                    concatenations of state variables"""
                )

    def set_state_vec_tol(self, state_vec, tol):
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
        self._atol[slices] = tol

    def _check_atol_type(self, size):
        """
        This method checks that the atol vector is of the right shape and
        type.

        Parameters
        ----------
        size: int
            The length of the atol vector
        """

        if isinstance(self._atol, float):
            self._atol = self._atol * np.ones(size)
        elif isinstance(self._atol, list):
            self._atol = np.array(self._atol)
        elif isinstance(self._atol, np.ndarray):
            pass
        else:
            raise pybamm.SolverError(
                "Absolute tolerances must be a numpy array, float, or list"
            )

        if self._atol.size != size:
            raise pybamm.SolverError(
                """Absolute tolerances must be either a scalar or a numpy arrray
                of the same shape at y0"""
            )

    def integrate(self, residuals, y0, t_eval, events, mass_matrix, jacobian):
        """
        Solve a DAE model defined by residuals with initial conditions y0.

        Parameters
        ----------
        residuals : method
            A function that takes in t, y and ydot and returns the residuals of the
            equations
        y0 : numeric type
            The initial conditions
        t_eval : numeric type
            The times at which to compute the solution
        events : method,
            A function that takes in t and y and returns conditions for the solver to
            stop
        mass_matrix : array_like,
            The (sparse) mass matrix for the chosen spatial method.
        jacobian : method,
            A function that takes in t and y and returns the Jacobian. If
            None, the solver will approximate the Jacobian.
            (see `SUNDIALS docs. <https://computation.llnl.gov/projects/sundials>`).
        """

        if jacobian is None:
            pybamm.SolverError("KLU requires the Jacobian to be provided")

        if events is None:
            pybamm.SolverError("KLU requires events to be provided")

        rtol = self._rtol
        self._check_atol_type(y0.size)
        atol = self._atol

        if jacobian:
            jac_y0_t0 = jacobian(t_eval[0], y0)
            if sparse.issparse(jac_y0_t0):

                def jacfn(t, y, cj):
                    j = jacobian(t, y) - cj * mass_matrix
                    return j

            else:

                def jacfn(t, y, cj):
                    jac_eval = jacobian(t, y) - cj * mass_matrix
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

        num_of_events = len(events)
        use_jac = 1

        def rootfn(t, y):
            return_root = np.ones((num_of_events,))
            return_root[:] = [event(t, y) for event in events]

            return return_root

        # get ids of rhs and algebraic variables
        rhs_ids = np.ones(self.rhs(0, y0).shape)
        alg_ids = np.zeros(self.algebraic(0, y0).shape)
        ids = np.concatenate((rhs_ids, alg_ids))

        # solve
        sol = idaklu.solve(
            t_eval,
            y0,
            ydot0,
            self.residuals,
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
            return pybamm.Solution(
                sol.t, np.transpose(y_out), t[-1], np.transpose(y_out[-1]), termination
            )
        else:
            raise pybamm.SolverError(sol.message)
