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


def _bdf_init(fun, jac, t0, y0, h0, rtol, atol):
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
    state = {}
    state['t'] = t0
    state['y'] = y0
    #state['fun'] = fun
    f0 = fun(t0, y0)
    state['atol'] = atol
    state['rtol'] = rtol
    order = 1
    state['order'] = order
    state['h'] = _select_initial_step(state, fun, t0, y0, f0, h0)
    EPS = np.finfo(y0.dtype).eps
    state['newton_tol'] = np.max((10 * EPS / rtol, np.min((0.03, rtol ** 0.5))))
    state['n_equal_steps'] = 0
    #state['jac'] = jac
    D = np.empty((MAX_ORDER + 1, len(y0)), dtype=y0.dtype)
    D = jax.ops.index_update(D, jax.ops.index[0, :], y0)
    D = jax.ops.index_update(D, jax.ops.index[1, :], f0 * h0)
    state['D'] = D
    state['y0'] = None
    state['scale_y0'] = None
    state = _predict(state)
    I = np.identity(len(y0), dtype=y0.dtype)
    state['I'] = I

    # kappa values for difference orders, taken from Table 1 of [1]
    kappa = np.array([0, -0.1850, -1 / 9, -0.0823, -0.0415, 0])
    gamma = np.hstack((0, np.cumsum(1 / np.arange(1, MAX_ORDER + 1))))
    alpha = 1.0 / ((1 - kappa) * gamma)
    c = h0 * alpha[order]
    error_const = kappa * gamma + 1 / np.arange(1, MAX_ORDER + 2)

    state['kappa'] = kappa
    state['gamma'] = gamma
    state['alpha'] = alpha
    state['c'] = c
    state['error_const'] = error_const

    J = jac(t0, y0)
    state['J'] = J
    state['LU'] = jax.scipy.linalg.lu_factor(I - c * J)

    state['U'] = [compute_R(order, 1) for order in range(MAX_ORDER)]
    state['psi'] = None
    state = _update_psi(state)
    return state


def _select_initial_step(state, fun, t0, y0, f0, h0):
    """
    Select a good initial step by stepping forward one step of forward euler, and
    comparing the predicted state against that using the provided function.

    Optimal step size based on the selected order is obtained using formula (4.12)
    in [4]
    """
    scale = state['atol'] + np.abs(y0) * state['rtol']
    y1 = y0 + h0 * f0
    f1 = fun(t0 + h0, y1)
    d2 = np.sqrt(np.mean(((f1 - f0) / scale)))
    order = 1
    h1 = h0 * d2 ** (-1 / (order + 1))
    return np.min((100 * h0, h1))


def _predict(state):
    """
    predict forward to new step (eq 2 in [1])
    """
    state['y0'] = np.sum(state['D'][:state['order'] + 1], axis=0)
    state['scale_y0'] = state['atol'] + state['rtol'] * np.abs(state['y0'])
    return state


def _update_psi(state):
    """
    update psi term as defined in second equation on page 9 of [1]
    """
    order = state['order']
    state['psi'] = np.dot(
        state['D'][1: order + 1].T,
        state['gamma'][1: order + 1]
    ) * state['alpha'][order]
    return state


def _update_difference_for_next_step(state, d, only_update_D=False):
    """
    update of difference equations can be done efficiently
    by reusing d and D.

    From first equation on page 4 of [1]:
    d = y_n - y^0_n = D^{k + 1} y_n

    Standard backwards difference gives
    D^{j + 1} y_n = D^{j} y_n - D^{j} y_{n - 1}

    Combining these gives the following algorithm
    """
    order = state['order']
    D = state['D']
    D = jax.ops.index_update(D, jax.ops.index[order + 2],
                             d - D[order + 1])
    D = jax.ops.index_update(D, jax.ops.index[order + 1],
                             d)
    for i in reversed(range(order + 1)):
        D = jax.ops.index_add(D, jax.ops.index[i],
                              D[i + 1])
    state['D'] = D

    if not only_update_D:
        # update psi (D has changed)
        state = _update_psi(state)

        # update y0 (D has changed)
        state = _predict(state)

    return state


def _update_step_size(state, factor, dont_update_lu=False):
    """
    If step size h is changed then also need to update the terms in
    the first equation of page 9 of [1]:

    - constant c = h / (1-kappa) gamma_k term
    - lu factorisation of (I - c * J) used in newton iteration (same equation)
    - psi term
    """
    order = state['order']
    h = state['h']

    h *= factor
    state['n_equal_steps'] = 0
    c = h * state['alpha'][order]

    # redo lu (c has changed)
    if not dont_update_lu:
        state['LU'] = scipy.linalg.lu_factor(state['I'] - c * state['J'])

    state['h'] = h
    state['c'] = c

    # update D using equations in section 3.2 of [1]
    RU = compute_R(order, factor).dot(state['U'][order])
    D = state['D']
    D = jax.ops.index_update(D, jax.ops.index[:order + 1],
                             np.dot(RU.T, D[:order + 1]))
    state['D'] = D

    # update psi (D has changed)
    state = _update_psi(state)

    # update y0 (D has changed)
    state = _predict(state)

    return state


def _update_jacobian(state, jac):
    """
    we update the jacobian using J(t_n, y_n) as per [1]

    Note: this is slightly different than the standard practice
    of using J(t_{n+1}, y^0_{n+1})
    """
    J = jac(state['t'], state['y'])
    state['LU'] = scipy.linalg.lu_factor(state['I'] - state['c'] * J)
    state['J'] = J
    return state


def _newton_iteration(state, fun):
    tol = state['newton_tol']
    c = state['c']
    psi = state['psi']
    y0 = state['y0']
    LU = state['LU']
    scale_y0 = state['scale_y0']
    t = state['t'] + state['h']
    d = np.zeros_like(y0)
    y = np.array(y0, copy=True)

    converged = False
    dy_norm_old = None
    k = 0
    while_state = [k, converged, dy_norm_old, d, y]

    def while_cond(state):
        k, converged, _, _, _ = state
        return not converged and k < NEWTON_MAXITER

    def while_body(state):
        k, converged, dy_norm_old, d, y = state
        f_eval = fun(t, y)
        b = c * f_eval - psi - d
        dy = jax.scipy.linalg.lu_solve(LU, b)
        dy_norm = np.sqrt(np.mean((dy / scale_y0)**2))
        if dy_norm_old is None:
            rate = None
        else:
            rate = dy_norm / dy_norm_old

        # if iteration is not going to converge in NEWTON_MAXITER
        # (assuming the current rate), then abort
        if (rate is not None and
                (rate >= 1 or
                 rate ** (NEWTON_MAXITER - k) / (1 - rate) * dy_norm > tol)):
            k = NEWTON_MAXITER
            return [k, converged, dy_norm_old, d, y]

        d += dy
        y = y0 + d

        # if converged then break out of iteration early
        if (dy_norm == 0 or
                rate is not None and rate / (1 - rate) * dy_norm < tol):
            converged = True

        dy_norm_old = dy_norm
        k += 1
        return [k, converged, dy_norm_old, d, y]

    k, converged, dy_norm_old, d, y = jax.lax.while_loop(while_cond, while_body,
                                                         while_state)

    return converged, k + 1, y, d


def _bdf_step(state, fun, jac):
    order = state['order']
    # we will try and use the old jacobian unless convergence of newton iteration
    # fails
    updated_jacobian = False
    # initialise step size and try to make the step,
    # iterate, reducing step size until error is in bounds
    step_accepted = False
    while not step_accepted:

        # solve BDF equation using y0 as starting point
        converged, n_iter, y, d = _newton_iteration(state, fun)

        # if not converged update the jacobian for J(t_n,y_n) and try again
        if not converged and not updated_jacobian:
            state = _update_jacobian(state, jac)
            updated_jacobian = True
            continue

        # if still not converged then multiply step size by 0.3 (as per [1])
        # and try again
        if not converged:
            state = _update_step_size(state, factor=0.3)
            continue

        # yay, converged, now check error is within bounds
        scale_y = state['atol'] + state['rtol'] * np.abs(y)

        # combine eq 3, 4 and 6 from [1] to obtain error
        # Note that error = C_k * h^{k+1} y^{k+1}
        # and d = D^{k+1} y_{n+1} \approx h^{k+1} y^{k+1}
        error = state['error_const'][order] * d
        error_norm = np.sqrt(np.mean((error / scale_y)**2))

        # calculate safety outside if since we will reuse later
        safety = 0.9 * (2 * NEWTON_MAXITER + 1) / (2 * NEWTON_MAXITER
                                                   + n_iter)

        # if error too large, reduce step size and try again
        if error_norm > 1:
            # calculate optimal step size factor as per eq 2.46 of [2]
            factor = np.max((MIN_FACTOR,
                             safety * error_norm ** (-1 / (order + 1))))
            state = _update_step_size(state, factor)
            continue

        # if we get here we can accept the step
        step_accepted = True

    # take the accepted step
    state['y'] = y
    state['t'] += state['h']

    # a change in order is only done after running at order k for k + 1 steps
    # (see page 83 of [2])
    state['n_equal_steps'] += 1
    if state['n_equal_steps'] < order + 1:
        state = _update_difference_for_next_step(state, d)
        return

    # don't need to update psi and y0 yet as we will be changing D again soon
    state = _update_difference_for_next_step(state, d, only_update_D=True)

    # similar to the optimal step size factor we calculated above for the current
    # order k, we need to calculate the optimal step size factors for orders
    # k-1 and k+1. To do this, we note that the error = C_k * D^{k+1} y_n
    if order > 1:
        error_m = state['error_const'][order - 1] * state['D'][order]
        error_m_norm = np.sqrt(np.mean((error_m / scale_y)**2))
    else:
        error_m_norm = np.inf

    if order < MAX_ORDER:
        error_p = state['error_const'][order + 1] * state['D'][order + 2]
        error_p_norm = np.sqrt(np.mean((error_p / scale_y)**2))
    else:
        error_p_norm = np.inf

    error_norms = np.array([error_m_norm, error_norm, error_p_norm])
    # with np.errstate(divide='ignore'):
    factors = error_norms ** (-1 / np.arange(order, order + 3))

    # now we have the three factors for orders k-1, k and k+1, pick the maximum in
    # order to maximise the resultant step size
    max_index = np.argmax(factors)
    order += max_index - 1
    state['order'] = order

    factor = np.min((MAX_FACTOR, safety * factors[max_index]))
    state = _update_step_size(state, factor)
    return state


def _bdf_interpolate(state, t_eval):
    """
    interpolate solution at time values t* where t-h < t* < t

    definition of the interpolating polynomial can be found on page 7 of [1]
    """
    order = state['order']
    t = state['t']
    h = state['h']
    D = state['D']
    time_steps = (t - h * np.arange(order)).reshape((-1, 1))
    denom = h * (1 + np.arange(order)).reshape((-1, 1))
    time_factor = np.cumprod((t_eval - time_steps) / denom, axis=0)

    order_summation = np.dot(D[1:order + 1].T, time_factor)
    return D[0].reshape(-1, 1) + order_summation


@jax.partial(jax.jit, static_argnums=(0, 1, 4, 5))
def _bdf_odeint(fun, jac, y0, t_eval, rtol, atol):
    t0 = t_eval[0]
    h0 = t_eval[1] - t0

    stepper = _bdf_init(fun, jac, t0, y0, h0, rtol, atol)
    i = 0
    y_out = np.empty((len(y0), len(t_eval)), dtype=y0.dtype)

    init_state = [stepper, t_eval, i, y_out]

    def cond_fun(state):
        _, t_eval, i, _ = state
        return i < len(t_eval)

    def body_fun(state):
        stepper, t_eval, i, y_out = state
        stepper = _bdf_step(stepper, fun, jac)
        index = np.searchsorted(t_eval, stepper.t)
        intermediate_times = t_eval[i:index]
        y_out = jax.ops.index_update(y_out, jax.ops.index[:, i:index],
                                     _bdf_interpolate(stepper, intermediate_times))
        i = index
        return [stepper, t_eval, i, y_out]

    stepper, t_eval, i, y_out = jax.lax.while_loop(cond_fun, body_fun, init_state)

    return y_out


def jax_bdf_integrate(jax_evaluate, y0, t_eval, rtol=1e-6, atol=1e-6):
    if not isinstance(jax_evaluate, pybamm.EvaluatorJax):
        raise ValueError("jax_evaluate must be an instance of pybamm.EvaluatorJax")

    @jax.jit
    def fun(t, y):
        return jax_evaluate.evaluate(t=t, y=y).reshape(-1)

    jac = jax.jacfwd(fun, argnums=1)

    y0_device = jax.device_put(y0).reshape(-1)
    t_eval_device = jax.device_put(t_eval)

    y_out = _bdf_odeint(fun, jac, y0_device, t_eval_device, rtol, atol)

    return np.array(y_out)
