import jax
import jax.numpy as np
import numpy as onp
import scipy
import pybamm

from jax.config import config
config.update("jax_enable_x64", True)



MAX_ORDER = 5
NEWTON_MAXITER = 4
MIN_FACTOR = 0.2
MAX_FACTOR = 10


def compute_R(order, factor):
    """
    computes the R matrix with entries
    given by the first equation on page 8 of [1]

    This is used to update the differences matrix when step size h is varied according
    to factor = h_{n+1} / h_n

    Note that the U matrix also defined in the same section can be also be
    found using factor = 1, which corresponds to R with a constant step size

    """
    I = np.arange(1, order + 1).reshape(-1, 1)
    J = np.arange(1, order + 1)
    M = np.zeros((order + 1, order + 1))
    M = jax.ops.index_update(M, jax.ops.index[1:, 1:], (I - 1 - factor * J) / I)
    M = jax.ops.index_update(M, jax.ops.index[0], 1)
    return np.cumprod(M, axis=0)


class BDF:
    """
    Backward Difference formula (BDF) implicit multistep integrator. The basic algorithm
    is derived in [2]. This particular implementation follows that implemented in the
    Matlab routine ode15s, described in [1], which features the NDF formulas for
    improved stability, with associated differences in the error constants, and
    calculates the jacobian at J(t_n, y_n), rather than at the more standard J(t_{n+1},
    y^0_{n+1}). This implementation was based on that implemented in the scipy library
    [3], which also mainly follows [1] but uses the more standard jacobian update.

    Parameters
    ----------

    fun: callable
        function with signature (t, y), where t is a scalar time and y is a ndarray with
        shape (n,), returns the rhs of the system of ODE equations as an nd array with
        shape (n,)
    jac: callable
        function with signature (t, y), where t is a scalar time and y is a ndarray with
        shape (n,), returns the jacobian matrix of fun as an ndarray with shape (n,n)
    t0: float
        initial time
    y0: ndarray
        initial state vector with shape (n,)


    References
    ----------
    .. [1] L. F. Shampine, M. W. Reichelt, "THE MATLAB ODE SUITE", SIAM J. SCI.
           COMPUTE., Vol. 18, No. 1, pp. 1-22, January 1997.
    .. [2] G. D. Byrne, A. C. Hindmarsh, "A Polyalgorithm for the Numerical
           Solution of Ordinary Differential Equations", ACM Transactions on
           Mathematical Software, Vol. 1, No. 1, pp. 71-96, March 1975.
    .. [3] Virtanen, P., Gommers, R., Oliphant, T. E., Haberland, M., Reddy,
           T., Cournapeau, D., ... & van der Walt, S. J. (2020). SciPy 1.0:
           fundamental algorithms for scientific computing in Python.
           Nature methods, 17(3), 261-272.
    .. [4] E. Hairer, S. P. Norsett G. Wanner, "Solving Ordinary Differential
               Equations I: Nonstiff Problems", Sec. II.4.
    """

    def __init__(self, fun, jac, t0, y0, h0, rtol, atol):
        self.t = t0
        self._y = y0
        # fun has signature f(t, y)
        self._fun = fun
        f0 = self._fun(self.t, self._y)
        self._atol = atol
        self._rtol = rtol
        self._order = 1
        self._h = self._select_initial_step(t0, y0, f0, h0)
        EPS = np.finfo(self._y.dtype).eps
        self._newton_tol = max(10 * EPS / self._rtol, min(0.03, self._rtol ** 0.5))
        self._n_equal_steps = 0
        self._jac = jac
        # Nordsieck differences array
        self._D = np.empty((MAX_ORDER + 1, len(y0)), dtype=self._y.dtype)
        self._D = jax.ops.index_update(self._D, jax.ops.index[0, :], self._y)
        self._D = jax.ops.index_update(self._D, jax.ops.index[1, :], f0 * self._h)
        self._y0 = None
        self._scale_y0 = None
        self._predict()

        self._I = np.identity(len(y0), dtype=self._y.dtype)

        # kappa values for difference orders, taken from Table 1 of [1]
        kappa = np.array([0, -0.1850, -1 / 9, -0.0823, -0.0415, 0])
        self._gamma = np.hstack((0, np.cumsum(1 / np.arange(1, MAX_ORDER + 1))))
        self._alpha = 1.0 / ((1 - kappa) * self._gamma)
        self._c = self._h * self._alpha[self._order]
        self._error_const = kappa * self._gamma + 1 / np.arange(1, MAX_ORDER + 2)

        self._J = self._jac(self.t, self._y)
        self._LU = scipy.linalg.lu_factor(self._I - self._c * self._J)

        self._U = [compute_R(order, 1) for order in range(MAX_ORDER)]

        self._psi = None
        self._update_psi()

    def _select_initial_step(self, t0, y0, f0, h0):
        """
        Select a good initial step by stepping forward one step of forward euler, and
        comparing the predicted state against that using the provided function.

        Optimal step size based on the selected order is obtained using formula (4.12)
        in [4]
        """
        scale = self._atol + np.abs(y0) * self._rtol
        y1 = y0 + h0 * f0
        f1 = self._fun(t0 + h0, y1)
        d2 = np.sqrt(np.mean(((f1 - f0) / scale)))
        order = 1
        h1 = h0 * d2 ** (-1 / (order + 1))

        return min(100 * h0, h1)

    def _predict(self):
        """
        predict forward to new step (eq 2 in [1])
        """
        self._y0 = np.sum(self._D[:self._order + 1], axis=0)
        self._scale_y0 = self._atol + self._rtol * np.abs(self._y0)

    def _update_psi(self):
        """
        update psi term as defined in second equation on page 9 of [1]
        """
        self._psi = np.dot(
            self._D[1: self._order + 1].T,
            self._gamma[1: self._order + 1]
        ) * self._alpha[self._order]

    def _update_difference_for_next_step(self, d, only_update_D=False):
        """
        update of difference equations can be done efficiently
        by reusing d and D.

        From first equation on page 4 of [1]:
        d = y_n - y^0_n = D^{k + 1} y_n

        Standard backwards difference gives
        D^{j + 1} y_n = D^{j} y_n - D^{j} y_{n - 1}

        Combining these gives the following algorithm
        """
        order = self._order
        self._D = jax.ops.index_update(self._D, jax.ops.index[order + 2],
                                       d - self._D[order + 1])
        self._D = jax.ops.index_update(self._D, jax.ops.index[order + 1],
                                       d)
        for i in reversed(range(order + 1)):
            self._D = jax.ops.index_add(self._D, jax.ops.index[i],
                                        self._D[i + 1])

        if not only_update_D:
            # update psi (D has changed)
            self._update_psi()

            # update y0 (D has changed)
            self._predict()

    def _update_step_size(self, factor, dont_update_lu=False):
        """
        If step size h is changed then also need to update the terms in
        the first equation of page 9 of [1]:

        - constant c = h / (1-kappa) gamma_k term
        - lu factorisation of (I - c * J) used in newton iteration (same equation)
        - psi term
        """
        self._h *= factor

        self._n_equal_steps = 0
        order = self._order
        self._c = self._h * self._alpha[order]

        # redo lu (c has changed)
        if not dont_update_lu:
            self._LU = scipy.linalg.lu_factor(self._I - self._c * self._J)

        # update D using equations in section 3.2 of [1]
        RU = compute_R(order, factor).dot(self._U[order])
        self._D = jax.ops.index_update(self._D, jax.ops.index[:order + 1],
                                       np.dot(RU.T, self._D[:order + 1]))

        # update psi (D has changed)
        self._update_psi()

        # update y0 (D has changed)
        self._predict()

    def _update_jacobian(self):
        """
        we update the jacobian using J(t_n, y_n) as per [1]

        Note: this is slightly different than the standard practice
        of using J(t_{n+1}, y^0_{n+1})
        """
        self._J = self._jac(self.t, self._y)
        self._LU = scipy.linalg.lu_factor(self._I - self._c * self._J)

    def _newton_iteration(self):
        tol = self._newton_tol
        t = self.t + self._h
        d = np.zeros_like(self._y0)
        y = self._y0.copy()

        converged = False
        dy_norm_old = None
        for k in range(NEWTON_MAXITER):
            f_eval = self._fun(t, y)
            b = self._c * f_eval - self._psi - d
            dy = scipy.linalg.lu_solve(self._LU, b)
            dy_norm = np.sqrt(np.mean((dy / self._scale_y0)**2))
            if dy_norm_old is None:
                rate = None
            else:
                rate = dy_norm / dy_norm_old

            # if iteration is not going to converge in NEWTON_MAXITER
            # (assuming the current rate), then abort
            if (rate is not None and
                    (rate >= 1 or
                     rate ** (NEWTON_MAXITER - k) / (1 - rate) * dy_norm > tol)):
                break
            d += dy
            y = self._y0 + d

            # if converged then break out of iteration early
            if (dy_norm == 0 or
                    rate is not None and rate / (1 - rate) * dy_norm < tol):
                converged = True
                break

        dy_norm_old = dy_norm

        return converged, k + 1, y, d

    def step(self):
        # we will try and use the old jacobian unless convergence of newton iteration
        # fails
        self._updated_jacobian = False
        # initialise step size and try to make the step,
        # iterate, reducing step size until error is in bounds
        step_accepted = False
        while not step_accepted:

            # solve BDF equation using self._y0 as starting point
            converged, n_iter, y, d = self._newton_iteration()

            # if not converged update the jacobian for J(t_n,y_n) and try again
            if not converged and not self._updated_jacobian:
                self._update_jacobian()
                self._updated_jacobian = True
                continue

            # if still not converged then multiply step size by 0.3 (as per [1])
            # and try again
            if not converged:
                self._update_step_size(factor=0.3)
                continue

            # yay, converged, now check error is within bounds
            scale_y = self._atol + self._rtol * np.abs(y)

            # combine eq 3, 4 and 6 from [1] to obtain error
            # Note that error = C_k * h^{k+1} y^{k+1}
            # and d = D^{k+1} y_{n+1} \approx h^{k+1} y^{k+1}
            error = self._error_const[self._order] * d
            error_norm = np.sqrt(np.mean((error / scale_y)**2))

            # calculate safety outside if since we will reuse later
            safety = 0.9 * (2 * NEWTON_MAXITER + 1) / (2 * NEWTON_MAXITER
                                                       + n_iter)

            # if error too large, reduce step size and try again
            if error_norm > 1:
                # calculate optimal step size factor as per eq 2.46 of [2]
                factor = max(MIN_FACTOR,
                             safety * error_norm ** (-1 / (self._order + 1)))
                self._update_step_size(factor)
                continue

            # if we get here we can accept the step
            step_accepted = True

        # take the accepted step
        self._y = y
        self.t += self._h

        # a change in order is only done after running at order k for k + 1 steps
        # (see page 83 of [2])
        self._n_equal_steps += 1
        if self._n_equal_steps < self._order + 1:
            self._update_difference_for_next_step(d)
            return True, None

        # don't need to update psi and y0 yet as we will be changing D again soon
        self._update_difference_for_next_step(d, only_update_D=True)

        # similar to the optimal step size factor we calculated above for the current
        # order k, we need to calculate the optimal step size factors for orders
        # k-1 and k+1. To do this, we note that the error = C_k * D^{k+1} y_n
        if self._order > 1:
            error_m = self._error_const[self._order - 1] * self._D[self._order]
            error_m_norm = np.sqrt(np.mean((error_m / scale_y)**2))
        else:
            error_m_norm = np.inf

        if self._order < MAX_ORDER:
            error_p = self._error_const[self._order + 1] * self._D[self._order + 2]
            error_p_norm = np.sqrt(np.mean((error_p / scale_y)**2))
        else:
            error_p_norm = np.inf

        error_norms = np.array([error_m_norm, error_norm, error_p_norm])
        # with np.errstate(divide='ignore'):
        factors = error_norms ** (-1 / np.arange(self._order, self._order + 3))

        # now we have the three factors for orders k-1, k and k+1, pick the maximum in
        # order to maximise the resultant step size
        max_index = np.argmax(factors)
        self._order += max_index - 1

        factor = min(MAX_FACTOR, safety * factors[max_index])
        self._update_step_size(factor)

    def interpolate(self, t_eval):
        """
        interpolate solution at time values t* where t-h < t* < t

        definition of the interpolating polynomial can be found on page 7 of [1]
        """
        time_steps = (self.t - self._h * np.arange(self._order)).reshape((-1, 1))
        denom = self._h * (1 + np.arange(self._order)).reshape((-1, 1))
        time_factor = np.cumprod((t_eval - time_steps) / denom, axis=0)

        order_summation = np.dot(self._D[1:self._order + 1].T, time_factor)
        return self._D[0].reshape(-1, 1) + order_summation


def jax_bdf_integrate(jax_evaluate, y0, t_eval, rtol=1e-6, atol=1e-6):
    if not isinstance(jax_evaluate, pybamm.EvaluatorJax):
        raise ValueError("jax_evaluate must be an instance of pybamm.EvaluatorJax")

    def fun(t, y):
        return jax_evaluate.evaluate(t=t, y=y).reshape(-1)

    jac = jax.jacfwd(fun, argnums=1)

    y0_device = jax.device_put(y0).reshape(-1)
    t_eval_device = jax.device_put(t_eval)

    t0 = t_eval_device[0]
    h0 = t_eval_device[1] - t0
    stepper = BDF(fun, jac, t0, y0_device, h0, rtol, atol)

    i = 0
    y_out = np.empty((len(y0), len(t_eval_device)), dtype=y0.dtype)
    while i < len(t_eval_device):
        stepper.step()
        index = np.searchsorted(t_eval_device, stepper.t)
        intermediate_times = t_eval_device[i:index]
        y_out = jax.ops.index_update(y_out, jax.ops.index[:, i:index],
                                     stepper.interpolate(intermediate_times))
        i = index
    return np.array(y_out)
