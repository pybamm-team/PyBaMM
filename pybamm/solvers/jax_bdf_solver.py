from functools import partial
import operator as op
import numpy as onp

import jax
import jax.numpy as jnp
from jax import core
from jax import dtypes
from jax.util import safe_map, safe_zip, cache, split_list
from jax.api_util import flatten_fun_nokwargs
from jax.flatten_util import ravel_pytree
from jax.tree_util import tree_map, tree_flatten, tree_unflatten
from jax.interpreters import partial_eval as pe
from jax import linear_util as lu

map = safe_map
zip = safe_zip


from jax.config import config
config.update("jax_enable_x64", True)


def jax_bdf_integrate(func, y0, t_eval, *args, rtol=1e-6, atol=1e-6):
    """
    Backward Difference formula (BDF) implicit multistep integrator. The basic algorithm
    is derived in [2]_. This particular implementation follows that implemented in the
    Matlab routine ode15s described in [1]_ and the SciPy implementation [3]_, which
    features the NDF formulas for improved stability, with associated differences in the
    error constants, and calculates the jacobian at J(t_{n+1}, y^0_{n+1}).  This
    implementation was based on that implemented in the scipy library [3]_, which also
    mainly follows [1]_ but uses the more standard jacobian update.

    Parameters
    ----------

    func: callable
        function to evaluate the time derivative of the solution `y` at time
        `t` as `func(y, t, *args)`, producing the same shape/structure as `y0`.
    y0: ndarray
        initial state vector
    t_eval: ndarray
        time points to evaluate the solution, has shape (m,)
    args: (optional)
        tuple of additional arguments for `fun`, which must be arrays
        scalars, or (nested) standard Python containers (tuples, lists, dicts,
        namedtuples, i.e. pytrees) of those types.
    rtol: (optional) float
        relative tolerance for the solver
    atol: (optional) float
        absolute tolerance for the solver

    Returns
    -------
    y: ndarray with shape (n, m)
        calculated state vector at each of the m time points

    stepper: dict
        internal variables of the stepper object

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
    """
    def _check_arg(arg):
        if not isinstance(arg, core.Tracer) and not core.valid_jaxtype(arg):
            msg = ("The contents of odeint *args must be arrays or scalars, but got "
                   "\n{}.")
        raise TypeError(msg.format(arg))

    flat_args, in_tree = tree_flatten((y0, t_eval[0], *args))
    in_avals = tuple(map(abstractify, flat_args))
    converted, consts = closure_convert(func, in_tree, in_avals)

    return _bdf_odeint_wrapper(converted, rtol, atol, y0, t_eval, *consts, *args)


MAX_ORDER = 5
NEWTON_MAXITER = 4
MIN_FACTOR = 0.2
MAX_FACTOR = 10


def flax_cond(pred, true_operand, true_fun,
              false_operand, false_fun):  # pragma: no cover
    """
    for debugging purposes, use this instead of jax.lax.cond
    """
    if pred:
        return true_fun(true_operand)
    else:
        return false_fun(false_operand)


def flax_while_loop(cond_fun, body_fun, init_val):  # pragma: no cover
    """
    for debugging purposes, use this instead of jax.lax.while_loop
    """
    val = init_val
    while cond_fun(val):
        val = body_fun(val)
    return val


def flax_fori_loop(start, stop, body_fun, init_val):  # pragma: no cover
    """
    for debugging purposes, use this instead of jax.lax.fori_loop
    """
    val = init_val
    for i in range(start, stop):
        val = body_fun(i, val)
    return val

def flax_scan(f, init, xs, length=None):  # pragma: no cover
    """
    for debugging purposes, use this instead of jax.lax.scan
    """
    if xs is None:
        xs = [None] * length
    carry = init
    ys = []
    for x in xs:
        carry, y = f(carry, x)
        ys.append(y)
    return carry, onp.stack(ys)


def _compute_R(order, factor):
    """
    computes the R matrix with entries
    given by the first equation on page 8 of [1]

    This is used to update the differences matrix when step size h is varied according
    to factor = h_{n+1} / h_n

    Note that the U matrix also defined in the same section can be also be
    found using factor = 1, which corresponds to R with a constant step size
    """
    I = jnp.arange(1, MAX_ORDER + 1).reshape(-1, 1)
    J = jnp.arange(1, MAX_ORDER + 1)
    M = jnp.empty((MAX_ORDER + 1, MAX_ORDER + 1))
    M = jax.ops.index_update(M, jax.ops.index[1:, 1:],
                             (I - 1 - factor * J) / I)
    M = jax.ops.index_update(M, jax.ops.index[0], 1)
    R = jnp.cumprod(M, axis=0)

    return R


def _bdf_init(fun, jac, t0, y0, h0, rtol, atol):
    """
    Initiation routine for Backward Difference formula (BDF) implicit multistep
    integrator.

    See jax_bdf_solver function above for details, this function returns a dict with the
    initial state of the solver

    Parameters
    ----------

    fun: callable
        function with signature (y, t), where t is a scalar time and y is a ndarray with
        shape (n,), returns the rhs of the system of ODE equations as an nd array with
        shape (n,)
    jac: callable
        function with signature (y, t), where t is a scalar time and y is a ndarray with
        shape (n,), returns the jacobian matrix of fun as an ndarray with shape (n,n)
    t0: float
        initial time
    y0: ndarray
        initial state vector with shape (n,)
    h0: float
        initial step size
    rtol: (optional) float
        relative tolerance for the solver
    atol: (optional) float
        absolute tolerance for the solver
    """
    state = {}
    state['t'] = t0
    state['y'] = y0
    f0 = fun(y0, t0)
    state['atol'] = atol
    state['rtol'] = rtol
    order = 1
    state['order'] = order
    state['h'] = _select_initial_step(state, fun, t0, y0, f0, h0)
    EPS = jnp.finfo(y0.dtype).eps
    state['newton_tol'] = jnp.max((10 * EPS / rtol, jnp.min((0.03, rtol ** 0.5))))
    state['n_equal_steps'] = 0
    D = jnp.empty((MAX_ORDER + 1, len(y0)), dtype=y0.dtype)
    D = jax.ops.index_update(D, jax.ops.index[0, :], y0)
    D = jax.ops.index_update(D, jax.ops.index[1, :], f0 * h0)
    state['D'] = D
    state['y0'] = None
    state['scale_y0'] = None
    state = _predict(state)
    I = jnp.identity(len(y0), dtype=y0.dtype)
    state['I'] = I

    # kappa values for difference orders, taken from Table 1 of [1]
    kappa = jnp.array([0, -0.1850, -1 / 9, -0.0823, -0.0415, 0])
    gamma = jnp.hstack((0, jnp.cumsum(1 / jnp.arange(1, MAX_ORDER + 1))))
    alpha = 1.0 / ((1 - kappa) * gamma)
    c = h0 * alpha[order]
    error_const = kappa * gamma + 1 / jnp.arange(1, MAX_ORDER + 2)

    state['kappa'] = kappa
    state['gamma'] = gamma
    state['alpha'] = alpha
    state['c'] = c
    state['error_const'] = error_const

    J = jac(y0, t0)
    state['J'] = J
    state['LU'] = jax.scipy.linalg.lu_factor(I - c * J)

    state['U'] = _compute_R(order, 1)
    state['psi'] = None
    state = _update_psi(state)

    state['n_function_evals'] = 2
    state['n_jacobian_evals'] = 1
    state['n_lu_decompositions'] = 1
    state['n_error_test_failures'] = 0
    return state


def _select_initial_step(state, fun, t0, y0, f0, h0):
    """
    Select a good initial step by stepping forward one step of forward euler, and
    comparing the predicted state against that using the provided function.

    Optimal step size based on the selected order is obtained using formula (4.12)
    in [1]

    References
    ----------
    .. [1] E. Hairer, S. P. Norsett G. Wanner, "Solving Ordinary Differential
               Equations I: Nonstiff Problems", Sec. II.4.
    """
    scale = state['atol'] + jnp.abs(y0) * state['rtol']
    y1 = y0 + h0 * f0
    f1 = fun(y1, t0 + h0)
    d2 = jnp.sqrt(jnp.mean(((f1 - f0) / scale)**2))
    order = 1
    h1 = h0 * d2 ** (-1 / (order + 1))
    return jnp.min((100 * h0, h1))


def _predict(state):
    """
    predict forward to new step (eq 2 in [1])
    """
    n = len(state['y'])
    order = state['order']
    orders = jnp.repeat(jnp.arange(MAX_ORDER + 1).reshape(-1, 1), n, axis=1)
    subD = jnp.where(orders <= order, state['D'], 0)
    state['y0'] = jnp.sum(subD, axis=0)
    state['scale_y0'] = state['atol'] + state['rtol'] * jnp.abs(state['y0'])
    return state


def _update_psi(state):
    """
    update psi term as defined in second equation on page 9 of [1]
    """
    order = state['order']
    n = len(state['y'])
    orders = jnp.arange(MAX_ORDER + 1)
    subGamma = jnp.where(orders > 0, jnp.where(orders <= order, state['gamma'], 0), 0)
    orders = jnp.repeat(orders.reshape(-1, 1), n, axis=1)
    subD = jnp.where(orders > 0, jnp.where(orders <= order, state['D'], 0), 0)
    state['psi'] = jnp.dot(
        subD.T,
        subGamma
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
    i = order
    while_state = [i, D]

    def while_cond(while_state):
        i, _ = while_state
        return i >= 0

    def while_body(while_state):
        i, D = while_state
        D = jax.ops.index_add(D, jax.ops.index[i],
                              D[i + 1])
        i -= 1
        return [i, D]

    i, D = jax.lax.while_loop(while_cond, while_body, while_state)

    state['D'] = D

    def update_psi_and_predict(state):
        # update psi (D has changed)
        state = _update_psi(state)

        # update y0 (D has changed)
        state = _predict(state)

        return state

    state = jax.lax.cond(only_update_D == False,  # noqa: E712
                         state, update_psi_and_predict,
                         state, lambda x: x)

    return state


def _update_step_size(state, factor, dont_update_lu):
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
    def update_lu(state):
        state['LU'] = jax.scipy.linalg.lu_factor(state['I'] - c * state['J'])
        state['n_lu_decompositions'] += 1
        return state

    state = jax.lax.cond(dont_update_lu == False,  # noqa: E712
                         state, update_lu,
                         state, lambda x: x)

    state['h'] = h
    state['c'] = c

    # update D using equations in section 3.2 of [1]
    RU = _compute_R(order, factor).dot(state['U'])
    I = jnp.arange(0, MAX_ORDER + 1).reshape(-1, 1)
    J = jnp.arange(0, MAX_ORDER + 1)

    # only update order+1, order+1 entries of D
    RU = jnp.where(jnp.logical_and(I <= order, J <= order),
                  RU, jnp.identity(MAX_ORDER + 1))
    D = state['D']
    D = jnp.dot(RU.T, D)
    # D = jax.ops.index_update(D, jax.ops.index[:order + 1],
    #                         jnp.dot(RU.T, D[:order + 1]))
    state['D'] = D

    # update psi (D has changed)
    state = _update_psi(state)

    # update y0 (D has changed)
    state = _predict(state)

    return state


def _update_jacobian(state, jac):
    """
    we update the jacobian using J(t_{n+1}, y^0_{n+1})
    following the scipy bdf implementation rather than J(t_n, y_n) as per [1]
    """
    J = jac(state['y0'], state['t'] + state['h'])
    state['n_jacobian_evals'] += 1
    state['LU'] = jax.scipy.linalg.lu_factor(state['I'] - state['c'] * J)
    state['n_lu_decompositions'] += 1
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
    d = jnp.zeros_like(y0)
    y = jnp.array(y0, copy=True)

    not_converged = True
    dy_norm_old = -1.0
    k = 0
    while_state = [k, not_converged, dy_norm_old, d, y, state]

    def while_cond(while_state):
        k, not_converged, _, _, _, _ = while_state
        return not_converged * (k < NEWTON_MAXITER)

    def while_body(while_state):
        k, not_converged, dy_norm_old, d, y, state = while_state
        f_eval = fun(y, t)
        state['n_function_evals'] += 1
        b = c * f_eval - psi - d
        dy = jax.scipy.linalg.lu_solve(LU, b)
        dy_norm = jnp.sqrt(jnp.mean((dy / scale_y0)**2))
        rate = dy_norm / dy_norm_old

        # if iteration is not going to converge in NEWTON_MAXITER
        # (assuming the current rate), then abort
        pred = rate >= 1
        pred += rate ** (NEWTON_MAXITER - k) / (1 - rate) * dy_norm > tol
        pred *= dy_norm_old >= 0
        k = jax.lax.cond(pred, k, lambda k: NEWTON_MAXITER, k, lambda k: k)

        d += dy
        y = y0 + d

        # if converged then break out of iteration early
        pred = dy_norm_old >= 0
        pred *= rate / (1 - rate) * dy_norm < tol
        pred += dy_norm == 0

        def converged_fun(not_converged):
            not_converged = False
            return not_converged

        def not_converged_fun(not_converged):
            return not_converged

        dy_norm_old = dy_norm

        not_converged = \
            jax.lax.cond(pred,
                         not_converged, converged_fun,
                         not_converged, not_converged_fun)
        return [k + 1, not_converged, dy_norm_old, d, y, state]

    k, not_converged, dy_norm_old, d, y, state = jax.lax.while_loop(while_cond,
                                                                    while_body,
                                                                    while_state)
    return not_converged, k, y, d, state


def _bdf_step(state, fun, jac):
    # we will try and use the old jacobian unless convergence of newton iteration
    # fails
    not_updated_jacobian = True
    # initialise step size and try to make the step,
    # iterate, reducing step size until error is in bounds
    step_accepted = False
    y = jnp.empty_like(state['y'])
    d = jnp.empty_like(state['y'])
    n_iter = -1

    # loop until step is accepted
    while_state = [state, step_accepted, not_updated_jacobian, y, d, n_iter]

    def while_cond(while_state):
        _, step_accepted, _, _, _, _ = while_state
        return step_accepted == False  # noqa: E712

    def while_body(while_state):
        state, step_accepted, not_updated_jacobian, y, d, n_iter = while_state

        # solve BDF equation using y0 as starting point
        not_converged, n_iter, y, d, state = _newton_iteration(state, fun)

        # if not converged update the jacobian for J(t_n,y_n) and try again
        pred = not_converged * not_updated_jacobian
        if_state = [state, not_updated_jacobian, step_accepted]

        def need_to_update_jacobian(if_state):
            # newton iteration did not converge, update the jacobian and try again
            state, not_updated_jacobian, step_accepted = if_state
            state = _update_jacobian(state, jac)
            not_updated_jacobian = False
            return [state, not_updated_jacobian, step_accepted]

        def dont_need_to_update_jacobian(if_state):
            state, not_updated_jacobian, step_accepted = if_state

            if_state2 = [state, step_accepted]

            def need_to_update_step_size(if_state2):
                # newton iteration did not converge, but jacobian has already been
                # evaluated so reduce step size by 0.3 (as per [1]) and try again
                state, step_accepted = if_state2
                state = _update_step_size(state, 0.3, False)
                return [state, step_accepted]

            def converged(if_state2):
                state, step_accepted = if_state2
                # yay, converged, now check error is within bounds
                scale_y = state['atol'] + state['rtol'] * jnp.abs(y)

                # combine eq 3, 4 and 6 from [1] to obtain error
                # Note that error = C_k * h^{k+1} y^{k+1}
                # and d = D^{k+1} y_{n+1} \approx h^{k+1} y^{k+1}
                error = state['error_const'][state['order']] * d
                error_norm = jnp.sqrt(jnp.mean((error / scale_y)**2))

                # calculate safety outside if since we will reuse later
                safety = 0.9 * (2 * NEWTON_MAXITER + 1) / (2 * NEWTON_MAXITER
                                                           + n_iter)

                if_state3 = [state, step_accepted]

                def error_too_large(if_state3):
                    # error too large, reduce step size and try again
                    state, step_accepted = if_state3
                    state['n_error_test_failures'] += 1
                    # calculate optimal step size factor as per eq 2.46 of [2]
                    factor = jnp.max((MIN_FACTOR,
                                     safety *
                                     error_norm ** (-1 / (state['order'] + 1))))
                    state = _update_step_size(state, factor, False)
                    return [state, step_accepted]

                def accept_step(if_state3):
                    # if we get here we can accept the step
                    state, step_accepted = if_state3
                    step_accepted = True
                    return [state, step_accepted]

                state, step_accepted = \
                    jax.lax.cond(error_norm > 1,
                                 if_state3, error_too_large,
                                 if_state3, accept_step)

                return [state, step_accepted]

            state, step_accepted = jax.lax.cond(not_converged,
                                                if_state2, need_to_update_step_size,
                                                if_state2, converged)

            return [state, not_updated_jacobian, step_accepted]

        state, not_updated_jacobian, step_accepted = \
            jax.lax.cond(pred,
                         if_state, need_to_update_jacobian,
                         if_state, dont_need_to_update_jacobian)
        return [state, step_accepted, not_updated_jacobian, y, d, n_iter]

    state, step_accepted, not_updated_jacobian, y, d, n_iter = \
        jax.lax.while_loop(while_cond, while_body, while_state)

    # take the accepted step
    state['y'] = y
    state['t'] += state['h']

    # a change in order is only done after running at order k for k + 1 steps
    # (see page 83 of [2])
    state['n_equal_steps'] += 1

    if_state = [state, d, y, n_iter]

    def no_change_in_order(if_state):
        # no change in order this step, update differences D and exit
        state, d, _, _ = if_state
        state = _update_difference_for_next_step(state, d, False)
        return state

    def order_change(if_state):
        state, d, y, n_iter = if_state
        order = state['order']

        # Note: we are recalculating these from the while loop above, could re-use?
        scale_y = state['atol'] + state['rtol'] * jnp.abs(y)
        error = state['error_const'][order] * d
        error_norm = jnp.sqrt(jnp.mean((error / scale_y)**2))
        safety = 0.9 * (2 * NEWTON_MAXITER + 1) / (2 * NEWTON_MAXITER
                                                   + n_iter)

        # don't need to update psi and y0 yet as we will be changing D again soon
        state = _update_difference_for_next_step(state, d, True)

        if_state2 = [state, scale_y, order]
        # similar to the optimal step size factor we calculated above for the current
        # order k, we need to calculate the optimal step size factors for orders
        # k-1 and k+1. To do this, we note that the error = C_k * D^{k+1} y_n

        def order_greater_one(if_state2):
            state, scale_y, order = if_state2
            error_m = state['error_const'][order - 1] * state['D'][order]
            error_m_norm = jnp.sqrt(jnp.mean((error_m / scale_y)**2))
            return error_m_norm

        def order_equal_one(if_state2):
            error_m_norm = jnp.inf
            return error_m_norm

        error_m_norm = jax.lax.cond(order > 1,
                                    if_state2, order_greater_one,
                                    if_state2, order_equal_one)

        def order_less_max(if_state2):
            state, scale_y, order = if_state2
            error_p = state['error_const'][order + 1] * state['D'][order + 2]
            error_p_norm = jnp.sqrt(jnp.mean((error_p / scale_y)**2))
            return error_p_norm

        def order_max(if_state2):
            error_p_norm = jnp.inf
            return error_p_norm

        error_p_norm = jax.lax.cond(order < MAX_ORDER,
                                    if_state2, order_less_max,
                                    if_state2, order_max)

        error_norms = jnp.array([error_m_norm, error_norm, error_p_norm])
        factors = error_norms ** (-1 / (jnp.arange(3) + order))

        # now we have the three factors for orders k-1, k and k+1, pick the maximum in
        # order to maximise the resultant step size
        max_index = jnp.argmax(factors)
        order += max_index - 1
        state['order'] = order

        factor = jnp.min((MAX_FACTOR, safety * factors[max_index]))
        state = _update_step_size(state, factor, False)

        return state

    state = jax.lax.cond(state['n_equal_steps'] < state['order'] + 1,
                         if_state, no_change_in_order,
                         if_state, order_change)

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
    j = 0
    time_factor = 1.0
    order_summation = D[0]
    while_state = [j, time_factor, order_summation]

    def while_cond(while_state):
        j, _, _ = while_state
        return j < order

    def while_body(while_state):
        j, time_factor, order_summation = while_state
        time_factor *= (t_eval - (t - h * j)) / (h * (1 + j))
        order_summation += D[j + 1] * time_factor
        j += 1
        return [j, time_factor, order_summation]

    j, time_factor, order_summation = jax.lax.while_loop(while_cond,
                                                         while_body,
                                                         while_state)
    return order_summation


@jax.partial(jax.custom_vjp, nondiff_argnums=(0, 1, 2))
def _bdf_odeint(fun, rtol, atol, y0, t_eval, *args):
    """
    main solver loop - creates a stepper object and steps through time, interpolating to
    the time points in t_eval
    """
    fun_bind_inputs = lambda y, t: fun(y, t, *args)

    jac_bind_inputs = jax.jacrev(fun_bind_inputs, argnums=0)

    t0 = t_eval[0]
    h0 = t_eval[1] - t0

    stepper = _bdf_init(fun_bind_inputs, jac_bind_inputs, t0, y0, h0, rtol, atol)
    i = 0
    y_out = jnp.empty((len(t_eval), len(y0)), dtype=y0.dtype)

    init_state = [stepper, t_eval, i, y_out, 0]

    def cond_fun(state):
        _, t_eval, i, _, _ = state
        return i < len(t_eval)

    def body_fun(state):
        stepper, t_eval, i, y_out, n_steps = state
        stepper = _bdf_step(stepper, fun_bind_inputs, jac_bind_inputs)
        index = jnp.searchsorted(t_eval, stepper['t'])

        def for_body(j, y_out):
            t = t_eval[j]
            y_out = jax.ops.index_update(y_out, jax.ops.index[j, :],
                                         _bdf_interpolate(stepper, t))
            return y_out

        y_out = jax.lax.fori_loop(i, index, for_body, y_out)
        return [stepper, t_eval, index, y_out, n_steps + 1]

    stepper, t_eval, i, y_out, n_steps = jax.lax.while_loop(cond_fun, body_fun,
                                                            init_state)

    stepper['n_steps'] = n_steps

    return y_out


@partial(jax.jit, static_argnums=(0, 1, 2))
def _bdf_odeint_wrapper(func, rtol, atol, y0, ts, *args):
    y0, unravel = ravel_pytree(y0)
    func = ravel_first_arg(func, unravel)
    out = _bdf_odeint(func, rtol, atol, y0, ts, *args)
    return jax.vmap(unravel)(out)


def _bdf_odeint_fwd(func, rtol, atol, y0, ts, *args):
    ys = _bdf_odeint(func, rtol, atol, y0, ts, *args)
    return ys, (ys, ts, args)


def _bdf_odeint_rev(func, rtol, atol, res, g):
    ys, ts, args = res

    def aug_dynamics(augmented_state, t, *args):
        """Original system augmented with vjp_y, vjp_t and vjp_args."""
        y, y_bar, *_ = augmented_state
        # `t` here is negative time, so we need to negate again to get back to
        # normal time. See the `odeint` invocation in `scan_fun` below.
        y_dot, vjpfun = jax.vjp(func, y, -t, *args)
        return (-y_dot, *vjpfun(y_bar))

    y_bar = g[-1]
    ts_bar = []
    t0_bar = 0.

    def scan_fun(carry, i):
        y_bar, t0_bar, args_bar = carry
        # Compute effect of moving measurement time
        t_bar = jnp.dot(func(ys[i], ts[i], *args), g[i])
        t0_bar = t0_bar - t_bar
        # Run augmented system backwards to previous observation
        _, y_bar, t0_bar, args_bar = jax_bdf_integrate(
            aug_dynamics, (ys[i], y_bar, t0_bar, args_bar),
            jnp.array([-ts[i], -ts[i - 1]]),
            *args, rtol=rtol, atol=atol)
        y_bar, t0_bar, args_bar = tree_map(op.itemgetter(1), (y_bar, t0_bar, args_bar))
        # Add gradient from current output
        y_bar = y_bar + g[i - 1]
        return (y_bar, t0_bar, args_bar), t_bar

    init_carry = (g[-1], 0., tree_map(jnp.zeros_like, args))
    (y_bar, t0_bar, args_bar), rev_ts_bar = jax.lax.scan(
        scan_fun, init_carry, jnp.arange(len(ts) - 1, 0, -1))
    ts_bar = jnp.concatenate([jnp.array([t0_bar]), rev_ts_bar[::-1]])
    return (y_bar, ts_bar, *args_bar)


_bdf_odeint.defvjp(_bdf_odeint_fwd, _bdf_odeint_rev)

@cache()
def closure_convert(fun, in_tree, in_avals):
    in_pvals = [pe.PartialVal.unknown(aval) for aval in in_avals]
    wrapped_fun, out_tree = flatten_fun_nokwargs(lu.wrap_init(fun), in_tree)
    with core.initial_style_staging():
        jaxpr, out_pvals, consts = pe.trace_to_jaxpr(
        wrapped_fun, in_pvals, instantiate=True, stage_out=False)
    out_tree = out_tree()

    # We only want to closure convert for constants with respect to which we're
    # differentiating. As a proxy for that, we hoist consts with float dtype.
    # TODO(mattjj): revise this approach
    is_float = lambda c: dtypes.issubdtype(dtypes.dtype(c), jnp.inexact)
    (closure_consts, hoisted_consts), merge = partition_list(is_float, consts)
    num_consts = len(hoisted_consts)

    def converted_fun(y, t, *hconsts_args):
        hoisted_consts, args = split_list(hconsts_args, [num_consts])
        consts = merge(closure_consts, hoisted_consts)
        all_args, in_tree2 = tree_flatten((y, t, *args))
        assert in_tree == in_tree2
        out_flat = core.eval_jaxpr(jaxpr, consts, *all_args)
        return tree_unflatten(out_tree, out_flat)

    return converted_fun, hoisted_consts

def partition_list(choice, lst):
    out = [], []
    which = [out[choice(elt)].append(elt) or choice(elt) for elt in lst]
    def merge(l1, l2):
        i1, i2 = iter(l1), iter(l2)
        return [next(i2 if snd else i1) for snd in which]
    return out, merge

def abstractify(x):
    return core.raise_to_shaped(core.get_aval(x))


def ravel_first_arg(f, unravel):
    return ravel_first_arg_(lu.wrap_init(f), unravel).call_wrapped

@lu.transformation
def ravel_first_arg_(unravel, y_flat, *args):
    y = unravel(y_flat)
    ans = yield (y,) + args, {}
    ans_flat, _ = ravel_pytree(ans)
    yield ans_flat
