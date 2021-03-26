import operator as op
import numpy as onp
import collections

import jax
import jax.numpy as jnp
from jax import core
from jax import dtypes
from jax.util import safe_map, cache, split_list
from jax.api_util import flatten_fun_nokwargs
from jax.flatten_util import ravel_pytree
from jax.tree_util import tree_map, tree_flatten, tree_unflatten, tree_multimap, partial
from jax.interpreters import partial_eval as pe
from jax import linear_util as lu
from jax.config import config
from absl import logging

logging.set_verbosity(logging.ERROR)

config.update("jax_enable_x64", True)

MAX_ORDER = 5
NEWTON_MAXITER = 4
ROOT_SOLVE_MAXITER = 15
MIN_FACTOR = 0.2
MAX_FACTOR = 10


@jax.partial(jax.custom_vjp, nondiff_argnums=(0, 1, 2, 3))
def _bdf_odeint(fun, mass, rtol, atol, y0, t_eval, *args):
    """
    This implements a Backward Difference formula (BDF) implicit multistep integrator.
    The basic algorithm is derived in [2]_. This particular implementation follows that
    implemented in the Matlab routine ode15s described in [1]_ and the SciPy
    implementation [3]_, which features the NDF formulas for improved stability, with
    associated differences in the error constants, and calculates the jacobian at
    J(t_{n+1}, y^0_{n+1}).  This implementation was based on that implemented in the
    scipy library [3]_, which also mainly follows [1]_ but uses the more standard
    jacobian update.

    Parameters
    ----------

    func: callable
        function to evaluate the time derivative of the solution `y` at time
        `t` as `func(y, t, *args)`, producing the same shape/structure as `y0`.
    mass: ndarray
        diagonal of the mass matrix with shape (n,)
    y0: ndarray
        initial state vector, has shape (n,)
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

    def fun_bind_inputs(y, t):
        return fun(y, t, *args)

    jac_bind_inputs = jax.jacfwd(fun_bind_inputs, argnums=0)

    t0 = t_eval[0]
    h0 = t_eval[1] - t0

    stepper = _bdf_init(fun_bind_inputs, jac_bind_inputs, mass, t0, y0, h0, rtol, atol)
    i = 0
    y_out = jnp.empty((len(t_eval), len(y0)), dtype=y0.dtype)

    init_state = [stepper, t_eval, i, y_out]

    def cond_fun(state):
        _, t_eval, i, _ = state
        return i < len(t_eval)

    def body_fun(state):
        stepper, t_eval, i, y_out = state
        stepper = _bdf_step(stepper, fun_bind_inputs, jac_bind_inputs)
        index = jnp.searchsorted(t_eval, stepper.t)

        def for_body(j, y_out):
            t = t_eval[j]
            y_out = jax.ops.index_update(
                y_out, jax.ops.index[j, :], _bdf_interpolate(stepper, t)
            )
            return y_out

        y_out = jax.lax.fori_loop(i, index, for_body, y_out)
        return [stepper, t_eval, index, y_out]

    stepper, t_eval, i, y_out = jax.lax.while_loop(cond_fun, body_fun, init_state)
    return y_out


BDFInternalStates = [
    "t",
    "atol",
    "rtol",
    "M",
    "newton_tol",
    "order",
    "h",
    "n_equal_steps",
    "D",
    "y0",
    "scale_y0",
    "kappa",
    "gamma",
    "alpha",
    "c",
    "error_const",
    "J",
    "LU",
    "U",
    "psi",
    "n_function_evals",
    "n_jacobian_evals",
    "n_lu_decompositions",
    "n_steps",
    "consistent_y0_failed",
]
BDFState = collections.namedtuple("BDFState", BDFInternalStates)

jax.tree_util.register_pytree_node(
    BDFState, lambda xs: (tuple(xs), None), lambda _, xs: BDFState(*xs)
)


def _bdf_init(fun, jac, mass, t0, y0, h0, rtol, atol):
    """
    Initiation routine for Backward Difference formula (BDF) implicit multistep
    integrator.

    See _bdf_odeint function above for details, this function returns a dict with the
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
    mass: ndarray
        diagonal of the mass matrix with shape (n,)
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
    state["t"] = t0
    state["atol"] = atol
    state["rtol"] = rtol
    state["M"] = mass
    EPS = jnp.finfo(y0.dtype).eps
    state["newton_tol"] = jnp.maximum(10 * EPS / rtol, jnp.minimum(0.03, rtol ** 0.5))

    scale_y0 = atol + rtol * jnp.abs(y0)
    y0, not_converged = _select_initial_conditions(
        fun, mass, t0, y0, state["newton_tol"], scale_y0
    )
    state["consistent_y0_failed"] = not_converged

    f0 = fun(y0, t0)
    order = 1
    state["order"] = order
    state["h"] = _select_initial_step(atol, rtol, fun, t0, y0, f0, h0)
    state["n_equal_steps"] = 0
    D = jnp.empty((MAX_ORDER + 1, len(y0)), dtype=y0.dtype)
    D = jax.ops.index_update(D, jax.ops.index[0, :], y0)
    D = jax.ops.index_update(D, jax.ops.index[1, :], f0 * state["h"])
    state["D"] = D
    state["y0"] = y0
    state["scale_y0"] = scale_y0

    # kappa values for difference orders, taken from Table 1 of [1]
    kappa = jnp.array([0, -0.1850, -1 / 9, -0.0823, -0.0415, 0])
    gamma = jnp.hstack((0, jnp.cumsum(1 / jnp.arange(1, MAX_ORDER + 1))))
    alpha = 1.0 / ((1 - kappa) * gamma)
    c = state["h"] * alpha[order]
    error_const = kappa * gamma + 1 / jnp.arange(1, MAX_ORDER + 2)

    state["kappa"] = kappa
    state["gamma"] = gamma
    state["alpha"] = alpha
    state["c"] = c
    state["error_const"] = error_const

    J = jac(y0, t0)
    state["J"] = J

    state["LU"] = jax.scipy.linalg.lu_factor(state["M"] - c * J)

    state["U"] = _compute_R(order, 1)
    state["psi"] = None

    state["n_function_evals"] = 2
    state["n_jacobian_evals"] = 1
    state["n_lu_decompositions"] = 1
    state["n_steps"] = 0

    tuple_state = BDFState(*[state[k] for k in BDFInternalStates])
    y0, scale_y0 = _predict(tuple_state, D)
    psi = _update_psi(tuple_state, D)
    return tuple_state._replace(y0=y0, scale_y0=scale_y0, psi=psi)


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
    M = jax.ops.index_update(M, jax.ops.index[1:, 1:], (I - 1 - factor * J) / I)
    M = jax.ops.index_update(M, jax.ops.index[0], 1)
    R = jnp.cumprod(M, axis=0)

    return R


def _select_initial_conditions(fun, M, t0, y0, tol, scale_y0):
    # identify algebraic variables as zeros on diagonal
    algebraic_variables = onp.diag(M) == 0.0

    # if all differentiable variables then return y0 (can use normal python if since M
    # is static)
    if not onp.any(algebraic_variables):
        return y0, False

    # calculate consistent initial conditions via a newton on -J_a @ delta = f_a This
    # follows this reference:
    #
    # Shampine, L. F., Reichelt, M. W., & Kierzenka, J. A. (1999). Solving index-1 DAEs
    # in MATLAB and Simulink. SIAM review, 41(3), 538-552.

    # calculate fun_a, function of algebraic variables
    def fun_a(y_a):
        y_full = jax.ops.index_update(y0, algebraic_variables, y_a)
        return fun(y_full, t0)[algebraic_variables]

    y0_a = y0[algebraic_variables]
    scale_y0_a = scale_y0[algebraic_variables]

    d = jnp.zeros(y0_a.shape[0], dtype=y0.dtype)
    y_a = jnp.array(y0_a, copy=True)

    # calculate neg jacobian of fun_a
    J_a = jax.jacfwd(fun_a)(y_a)
    LU = jax.scipy.linalg.lu_factor(-J_a)

    converged = False
    dy_norm_old = -1.0
    k = 0
    while_state = [k, converged, dy_norm_old, d, y_a]

    def while_cond(while_state):
        k, converged, _, _, _ = while_state
        return (converged == False) * (k < ROOT_SOLVE_MAXITER)  # noqa: E712

    def while_body(while_state):
        k, converged, dy_norm_old, d, y_a = while_state
        f_eval = fun_a(y_a)
        dy = jax.scipy.linalg.lu_solve(LU, f_eval)
        dy_norm = jnp.sqrt(jnp.mean((dy / scale_y0_a) ** 2))
        rate = dy_norm / dy_norm_old

        d += dy
        y_a = y0_a + d

        # if converged then break out of iteration early
        pred = dy_norm_old >= 0.0
        pred *= rate / (1 - rate) * dy_norm < tol
        converged = (dy_norm == 0.0) + pred

        dy_norm_old = dy_norm

        return [k + 1, converged, dy_norm_old, d, y_a]

    k, converged, dy_norm_old, d, y_a = jax.lax.while_loop(
        while_cond, while_body, while_state
    )
    y_tilde = jax.ops.index_update(y0, algebraic_variables, y_a)

    return y_tilde, converged


def _select_initial_step(atol, rtol, fun, t0, y0, f0, h0):
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
    scale = atol + jnp.abs(y0) * rtol
    y1 = y0 + h0 * f0
    f1 = fun(y1, t0 + h0)
    d2 = jnp.sqrt(jnp.mean(((f1 - f0) / scale) ** 2))
    order = 1
    h1 = h0 * d2 ** (-1 / (order + 1))
    return jnp.minimum(100 * h0, h1)


def _predict(state, D):
    """
    predict forward to new step (eq 2 in [1])
    """
    n = len(state.y0)
    order = state.order
    orders = jnp.repeat(jnp.arange(MAX_ORDER + 1).reshape(-1, 1), n, axis=1)
    subD = jnp.where(orders <= order, D, 0)
    y0 = jnp.sum(subD, axis=0)
    scale_y0 = state.atol + state.rtol * jnp.abs(state.y0)
    return y0, scale_y0


def _update_psi(state, D):
    """
    update psi term as defined in second equation on page 9 of [1]
    """
    order = state.order
    n = len(state.y0)
    orders = jnp.arange(MAX_ORDER + 1)
    subGamma = jnp.where(orders > 0, jnp.where(orders <= order, state.gamma, 0), 0)
    orders = jnp.repeat(orders.reshape(-1, 1), n, axis=1)
    subD = jnp.where(orders > 0, jnp.where(orders <= order, D, 0), 0)
    psi = jnp.dot(subD.T, subGamma) * state.alpha[order]
    return psi


def _update_difference_for_next_step(state, d):
    """
    update of difference equations can be done efficiently
    by reusing d and D.

    From first equation on page 4 of [1]:
    d = y_n - y^0_n = D^{k + 1} y_n

    Standard backwards difference gives
    D^{j + 1} y_n = D^{j} y_n - D^{j} y_{n - 1}

    Combining these gives the following algorithm
    """
    order = state.order
    D = state.D
    D = jax.ops.index_update(D, jax.ops.index[order + 2], d - D[order + 1])
    D = jax.ops.index_update(D, jax.ops.index[order + 1], d)
    i = order
    while_state = [i, D]

    def while_cond(while_state):
        i, _ = while_state
        return i >= 0

    def while_body(while_state):
        i, D = while_state
        D = jax.ops.index_add(D, jax.ops.index[i], D[i + 1])
        i -= 1
        return [i, D]

    i, D = jax.lax.while_loop(while_cond, while_body, while_state)

    return D


def _update_step_size_and_lu(state, factor):
    state = _update_step_size(state, factor)

    # redo lu (c has changed)
    LU = jax.scipy.linalg.lu_factor(state.M - state.c * state.J)
    n_lu_decompositions = state.n_lu_decompositions + 1

    return state._replace(LU=LU, n_lu_decompositions=n_lu_decompositions)


def _update_step_size(state, factor):
    """
    If step size h is changed then also need to update the terms in
    the first equation of page 9 of [1]:

    - constant c = h / (1-kappa) gamma_k term
    - lu factorisation of (M - c * J) used in newton iteration (same equation)
    - psi term
    """
    order = state.order
    h = state.h * factor
    n_equal_steps = 0
    c = h * state.alpha[order]

    # update D using equations in section 3.2 of [1]
    RU = _compute_R(order, factor).dot(state.U)
    I = jnp.arange(0, MAX_ORDER + 1).reshape(-1, 1)
    J = jnp.arange(0, MAX_ORDER + 1)

    # only update order+1, order+1 entries of D
    RU = jnp.where(
        jnp.logical_and(I <= order, J <= order), RU, jnp.identity(MAX_ORDER + 1)
    )
    D = state.D
    D = jnp.dot(RU.T, D)
    # D = jax.ops.index_update(D, jax.ops.index[:order + 1],
    #                         jnp.dot(RU.T, D[:order + 1]))

    # update psi (D has changed)
    psi = _update_psi(state, D)

    # update y0 (D has changed)
    y0, scale_y0 = _predict(state, D)

    return state._replace(
        n_equal_steps=n_equal_steps, h=h, c=c, D=D, psi=psi, y0=y0, scale_y0=scale_y0
    )


def _update_jacobian(state, jac):
    """
    we update the jacobian using J(t_{n+1}, y^0_{n+1})
    following the scipy bdf implementation rather than J(t_n, y_n) as per [1]
    """
    J = jac(state.y0, state.t + state.h)
    n_jacobian_evals = state.n_jacobian_evals + 1
    LU = jax.scipy.linalg.lu_factor(state.M - state.c * J)
    n_lu_decompositions = state.n_lu_decompositions + 1
    return state._replace(
        J=J,
        n_jacobian_evals=n_jacobian_evals,
        LU=LU,
        n_lu_decompositions=n_lu_decompositions,
    )


def _newton_iteration(state, fun):
    tol = state.newton_tol
    c = state.c
    psi = state.psi
    y0 = state.y0
    LU = state.LU
    M = state.M
    scale_y0 = state.scale_y0
    t = state.t + state.h
    d = jnp.zeros(y0.shape, dtype=y0.dtype)
    y = jnp.array(y0, copy=True)
    n_function_evals = state.n_function_evals

    converged = False
    dy_norm_old = -1.0
    k = 0
    while_state = [k, converged, dy_norm_old, d, y, n_function_evals]

    def while_cond(while_state):
        k, converged, _, _, _, _ = while_state
        return (converged == False) * (k < NEWTON_MAXITER)  # noqa: E712

    def while_body(while_state):
        k, converged, dy_norm_old, d, y, n_function_evals = while_state
        f_eval = fun(y, t)
        n_function_evals += 1
        b = c * f_eval - M @ (psi + d)
        dy = jax.scipy.linalg.lu_solve(LU, b)
        dy_norm = jnp.sqrt(jnp.mean((dy / scale_y0) ** 2))
        rate = dy_norm / dy_norm_old

        # if iteration is not going to converge in NEWTON_MAXITER
        # (assuming the current rate), then abort
        pred = rate >= 1
        pred += rate ** (NEWTON_MAXITER - k) / (1 - rate) * dy_norm > tol
        pred *= dy_norm_old >= 0
        k += pred * (NEWTON_MAXITER - k - 1)

        d += dy
        y = y0 + d

        # if converged then break out of iteration early
        pred = dy_norm_old >= 0.0
        pred *= rate / (1 - rate) * dy_norm < tol
        converged = (dy_norm == 0.0) + pred

        dy_norm_old = dy_norm

        return [k + 1, converged, dy_norm_old, d, y, n_function_evals]

    k, converged, dy_norm_old, d, y, n_function_evals = jax.lax.while_loop(
        while_cond, while_body, while_state
    )
    return converged, k, y, d, state._replace(n_function_evals=n_function_evals)


def rms_norm(arg):
    return jnp.sqrt(jnp.mean(arg ** 2))


def _prepare_next_step(state, d):
    D = _update_difference_for_next_step(state, d)
    psi = _update_psi(state, D)
    y0, scale_y0 = _predict(state, D)
    return state._replace(D=D, psi=psi, y0=y0, scale_y0=scale_y0)


def _prepare_next_step_order_change(state, d, y, n_iter):
    order = state.order

    D = _update_difference_for_next_step(state, d)

    # Note: we are recalculating these from the while loop above, could re-use?
    scale_y = state.atol + state.rtol * jnp.abs(y)
    error = state.error_const[order] * d
    error_norm = rms_norm(error / scale_y)
    safety = 0.9 * (2 * NEWTON_MAXITER + 1) / (2 * NEWTON_MAXITER + n_iter)

    # similar to the optimal step size factor we calculated above for the current
    # order k, we need to calculate the optimal step size factors for orders
    # k-1 and k+1. To do this, we note that the error = C_k * D^{k+1} y_n
    error_m_norm = jnp.where(
        order > 1, rms_norm(state.error_const[order - 1] * D[order] / scale_y), jnp.inf
    )
    error_p_norm = jnp.where(
        order < MAX_ORDER,
        rms_norm(state.error_const[order + 1] * D[order + 2] / scale_y),
        jnp.inf,
    )

    error_norms = jnp.array([error_m_norm, error_norm, error_p_norm])
    factors = error_norms ** (-1 / (jnp.arange(3) + order))

    # now we have the three factors for orders k-1, k and k+1, pick the maximum in
    # order to maximise the resultant step size
    max_index = jnp.argmax(factors)
    order += max_index - 1

    factor = jnp.minimum(MAX_FACTOR, safety * factors[max_index])

    new_state = _update_step_size_and_lu(state._replace(D=D, order=order), factor)
    return new_state


def _bdf_step(state, fun, jac):
    # print('bdf_step', state.t, state.h)
    # we will try and use the old jacobian unless convergence of newton iteration
    # fails
    updated_jacobian = False
    # initialise step size and try to make the step,
    # iterate, reducing step size until error is in bounds
    step_accepted = False
    y = jnp.empty_like(state.y0)
    d = jnp.empty_like(state.y0)
    n_iter = -1

    # loop until step is accepted
    while_state = [state, step_accepted, updated_jacobian, y, d, n_iter]

    def while_cond(while_state):
        _, step_accepted, _, _, _, _ = while_state
        return step_accepted == False  # noqa: E712

    def while_body(while_state):
        state, step_accepted, updated_jacobian, y, d, n_iter = while_state

        # solve BDF equation using y0 as starting point
        converged, n_iter, y, d, state = _newton_iteration(state, fun)
        not_converged = converged == False  # noqa: E712

        # newton iteration did not converge, but jacobian has already been
        # evaluated so reduce step size by 0.3 (as per [1]) and try again
        state = tree_multimap(
            partial(jnp.where, not_converged * updated_jacobian),
            _update_step_size_and_lu(state, 0.3),
            state,
        )

        # if not_converged * updated_jacobian:
        #    print('not converged, update step size by 0.3')
        # if not_converged * (updated_jacobian == False):
        #    print('not converged, update jacobian')

        # if not converged and jacobian not updated, then update the jacobian and try
        # again
        (state, updated_jacobian) = tree_multimap(
            partial(
                jnp.where, not_converged * (updated_jacobian == False)  # noqa: E712
            ),
            (_update_jacobian(state, jac), True),
            (state, False + updated_jacobian),
        )

        safety = 0.9 * (2 * NEWTON_MAXITER + 1) / (2 * NEWTON_MAXITER + n_iter)
        scale_y = state.atol + state.rtol * jnp.abs(y)

        # combine eq 3, 4 and 6 from [1] to obtain error
        # Note that error = C_k * h^{k+1} y^{k+1}
        # and d = D^{k+1} y_{n+1} \approx h^{k+1} y^{k+1}
        error = state.error_const[state.order] * d

        error_norm = rms_norm(error / scale_y)

        # calculate optimal step size factor as per eq 2.46 of [2]
        factor = jnp.maximum(
            MIN_FACTOR, safety * error_norm ** (-1 / (state.order + 1))
        )

        # if converged * (error_norm > 1):
        #    print('converged, but error is too large',error_norm, factor, d, scale_y)

        (state, step_accepted) = tree_multimap(
            partial(jnp.where, converged * (error_norm > 1)),  # noqa: E712
            (_update_step_size_and_lu(state, factor), False),
            (state, converged),
        )

        return [state, step_accepted, updated_jacobian, y, d, n_iter]

    state, step_accepted, updated_jacobian, y, d, n_iter = jax.lax.while_loop(
        while_cond, while_body, while_state
    )

    # take the accepted step
    n_steps = state.n_steps + 1
    t = state.t + state.h

    # a change in order is only done after running at order k for k + 1 steps
    # (see page 83 of [2])
    n_equal_steps = state.n_equal_steps + 1

    state = state._replace(n_equal_steps=n_equal_steps, t=t, n_steps=n_steps)

    state = tree_multimap(
        partial(jnp.where, n_equal_steps < state.order + 1),
        _prepare_next_step(state, d),
        _prepare_next_step_order_change(state, d, y, n_iter),
    )

    return state


def _bdf_interpolate(state, t_eval):
    """
    interpolate solution at time values t* where t-h < t* < t

    definition of the interpolating polynomial can be found on page 7 of [1]
    """
    order = state.order
    t = state.t
    h = state.h
    D = state.D
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

    j, time_factor, order_summation = jax.lax.while_loop(
        while_cond, while_body, while_state
    )
    return order_summation


def block_diag(lst):
    def block_fun(i, j, Ai, Aj):
        if i == j:
            return Ai
        else:
            return onp.zeros(
                (
                    Ai.shape[0] if Ai.ndim > 1 else 1,
                    Aj.shape[1] if Aj.ndim > 1 else 1,
                ),
                dtype=Ai.dtype,
            )

    blocks = [
        [block_fun(i, j, Ai, Aj) for j, Aj in enumerate(lst)]
        for i, Ai in enumerate(lst)
    ]

    return onp.block(blocks)


# NOTE: the code below (except the docstring on jax_bdf_integrate and other minor
# edits), has been modified from the JAX library at https://github.com/google/jax.
# The main difference is the addition of support for semi-explicit dae index 1 problems
# via the addition of a mass matrix.
# This is under an Apache license, a short form of which is given here:
#
# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this
# file except in compliance with the License.  You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under
# the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied.  See the License for the specific language
# governing permissions and limitations under the License.


def jax_bdf_integrate(func, y0, t_eval, *args, rtol=1e-6, atol=1e-6, mass=None):
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
    mass: (optional) ndarray
        diagonal of the mass matrix with shape (n,)

    Returns
    -------
    y: ndarray with shape (n, m)
        calculated state vector at each of the m time points

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
            msg = (
                "The contents of odeint *args must be arrays or scalars, but got "
                "\n{}."
            )
        raise TypeError(msg.format(arg))

    flat_args, in_tree = tree_flatten((y0, t_eval[0], *args))
    in_avals = tuple(safe_map(abstractify, flat_args))
    converted, consts = closure_convert(func, in_tree, in_avals)
    return _bdf_odeint_wrapper(converted, mass, rtol, atol, y0, t_eval, *consts, *args)


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


@jax.partial(jax.jit, static_argnums=(0, 1, 2, 3))
def _bdf_odeint_wrapper(func, mass, rtol, atol, y0, ts, *args):
    y0, unravel = ravel_pytree(y0)
    if mass is None:
        mass = onp.identity(y0.shape[0], dtype=y0.dtype)
    else:
        mass = block_diag(tree_flatten(mass)[0])
    func = ravel_first_arg(func, unravel)
    out = _bdf_odeint(func, mass, rtol, atol, y0, ts, *args)
    return jax.vmap(unravel)(out)


def _bdf_odeint_fwd(func, mass, rtol, atol, y0, ts, *args):
    ys = _bdf_odeint(func, mass, rtol, atol, y0, ts, *args)
    return ys, (ys, ts, args)


def _bdf_odeint_rev(func, mass, rtol, atol, res, g):
    ys, ts, args = res

    def aug_dynamics(augmented_state, t, *args):
        """Original system augmented with vjp_y, vjp_t and vjp_args."""
        y, y_bar, *_ = augmented_state
        # `t` here is negative time, so we need to negate again to get back to
        # normal time. See the `odeint` invocation in `scan_fun` below.
        y_dot, vjpfun = jax.vjp(func, y, -t, *args)

        # Adjoint equations for semi-explicit dae index 1 system from
        #
        # [1] Cao, Y., Li, S., Petzold, L., & Serban, R. (2003). Adjoint sensitivity
        # analysis for differential-algebraic equations: The adjoint DAE system and its
        # numerical solution. SIAM journal on scientific computing, 24(3), 1076-1089.
        #
        # y_bar_dot_d = -J_dd^T y_bar_d - J_ad^T y_bar_a
        #           0 =  J_da^T y_bar_d + J_aa^T y_bar_d

        y_bar_dot, *rest = vjpfun(y_bar)

        return (-y_dot, y_bar_dot, *rest)

    algebraic_variables = onp.diag(mass) == 0.0
    differentiable_variables = algebraic_variables == False  # noqa: E712
    mass_is_I = onp.array_equal(mass, onp.eye(mass.shape[0]))
    is_dae = onp.any(algebraic_variables)

    if not mass_is_I:
        M_dd = mass[onp.ix_(differentiable_variables, differentiable_variables)]
        LU_invM_dd = jax.scipy.linalg.lu_factor(M_dd)

    def initialise(g0, y0, t0):
        # [1] gives init conditions for y_bar_a = g_d - J_ad^T (J_aa^T)^-1 g_a
        if mass_is_I:
            y_bar = g0
        elif is_dae:
            J = jax.jacfwd(func)(y0, t0, *args)

            # boolean arguments not implemented in jnp.ix_
            J_aa = J[onp.ix_(algebraic_variables, algebraic_variables)]
            J_ad = J[onp.ix_(algebraic_variables, differentiable_variables)]
            LU = jax.scipy.linalg.lu_factor(J_aa)
            g0_a = g0[algebraic_variables]
            invJ_aa = jax.scipy.linalg.lu_solve(LU, g0_a)
            y_bar = jax.ops.index_update(
                g0,
                differentiable_variables,
                jax.scipy.linalg.lu_solve(LU_invM_dd, g0_a - J_ad @ invJ_aa),
            )
        else:
            y_bar = jax.scipy.linalg.lu_solve(LU_invM_dd, g0)
        return y_bar

    y_bar = initialise(g[-1], ys[-1], ts[-1])
    ts_bar = []
    t0_bar = 0.0

    def arg_to_identity(arg):
        return onp.identity(arg.shape[0] if arg.ndim > 0 else 1, dtype=arg.dtype)

    def arg_dicts_to_values(args):
        """
        Note: JAX puts in empty arrays into args for some reason, we remove them here
        """
        return sum((tuple(b.values()) for b in args if isinstance(b, dict)), ())

    aug_mass = (mass, mass, onp.array(1.0)) + arg_dicts_to_values(
        tree_map(arg_to_identity, args)
    )

    def scan_fun(carry, i):
        y_bar, t0_bar, args_bar = carry
        # Compute effect of moving measurement time
        t_bar = jnp.dot(func(ys[i], ts[i], *args), g[i])
        t0_bar = t0_bar - t_bar
        # Run augmented system backwards to previous observation
        _, y_bar, t0_bar, args_bar = jax_bdf_integrate(
            aug_dynamics,
            (ys[i], y_bar, t0_bar, args_bar),
            jnp.array([-ts[i], -ts[i - 1]]),
            *args,
            mass=aug_mass,
            rtol=rtol,
            atol=atol,
        )
        y_bar, t0_bar, args_bar = tree_map(op.itemgetter(1), (y_bar, t0_bar, args_bar))
        # Add gradient from current output
        y_bar = y_bar + initialise(g[i - 1], ys[i - 1], ts[i - 1])
        return (y_bar, t0_bar, args_bar), t_bar

    init_carry = (y_bar, t0_bar, tree_map(jnp.zeros_like, args))
    (y_bar, t0_bar, args_bar), rev_ts_bar = jax.lax.scan(
        scan_fun, init_carry, jnp.arange(len(ts) - 1, 0, -1)
    )
    ts_bar = jnp.concatenate([jnp.array([t0_bar]), rev_ts_bar[::-1]])
    return (y_bar, ts_bar, *args_bar)


_bdf_odeint.defvjp(_bdf_odeint_fwd, _bdf_odeint_rev)


@cache()
def closure_convert(fun, in_tree, in_avals):
    if config.omnistaging_enabled:
        wrapped_fun, out_tree = flatten_fun_nokwargs(lu.wrap_init(fun), in_tree)
        jaxpr, _, consts = pe.trace_to_jaxpr_dynamic(wrapped_fun, in_avals)
    else:
        in_pvals = [pe.PartialVal.unknown(aval) for aval in in_avals]
        wrapped_fun, out_tree = flatten_fun_nokwargs(lu.wrap_init(fun), in_tree)
        with core.initial_style_staging():  # type: ignore
            jaxpr, _, consts = pe.trace_to_jaxpr(
                wrapped_fun, in_pvals, instantiate=True, stage_out=False
            )  # type: ignore
    out_tree = out_tree()

    # We only want to closure convert for constants with respect to which we're
    # differentiating. As a proxy for that, we hoist consts with float dtype.
    # TODO(mattjj): revise this approach
    def is_float(c):
        return dtypes.issubdtype(dtypes.dtype(c), jnp.inexact)

    (closure_consts, hoisted_consts), merge = partition_list(is_float, consts)
    num_consts = len(hoisted_consts)

    def converted_fun(y, t, *hconsts_args):
        hoisted_consts, args = split_list(hconsts_args, [num_consts])
        consts = merge(closure_consts, hoisted_consts)
        all_args, _ = tree_flatten((y, t, *args))
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
