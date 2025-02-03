# mypy: ignore-errors
import collections
import operator as op
from functools import partial

import numpy as onp

import pybamm

if pybamm.has_jax():
    import jax
    import jax.numpy as jnp
    from jax import core, dtypes
    from jax.extend import linear_util as lu
    from jax.api_util import flatten_fun_nokwargs
    from jax.flatten_util import ravel_pytree
    from jax.interpreters import partial_eval as pe
    from jax.tree_util import tree_flatten, tree_map, tree_unflatten
    from jax.util import cache, safe_map, split_list

    platform = jax.lib.xla_bridge.get_backend().platform.casefold()
    if platform != "metal":
        jax.config.update("jax_enable_x64", True)

    MAX_ORDER = 5
    NEWTON_MAXITER = 4
    ROOT_SOLVE_MAXITER = 15
    MIN_FACTOR = 0.2
    MAX_FACTOR = 10

    # https://github.com/jax-ml/jax/issues/4572#issuecomment-709809897
    def some_hash_function(x):
        return hash(str(x))

    class HashableArrayWrapper:
        """wrapper for a numpy array to make it hashable"""

        def __init__(self, val):
            self.val = val

        def __hash__(self):
            return some_hash_function(self.val)

        def __eq__(self, other):
            return isinstance(other, HashableArrayWrapper) and onp.all(
                onp.equal(self.val, other.val)
            )

    def gnool_jit(fun, static_array_argnums=(), static_argnums=()):
        """redefinition of jax jit to allow static array args"""

        @partial(jax.jit, static_argnums=static_array_argnums)
        def callee(*args):
            args = list(args)
            for i in static_array_argnums:
                args[i] = args[i].val
            return fun(*args)

        def caller(*args):
            args = list(args)
            for i in static_array_argnums:
                args[i] = HashableArrayWrapper(args[i])
            return callee(*args)

        return caller

    @partial(jax.custom_vjp, nondiff_argnums=(0, 1, 2, 3))
    def _bdf_odeint(fun, mass, rtol, atol, y0, t_eval, *args):
        """
        Implements a Backward Difference formula (BDF) implicit multistep integrator.
        The basic algorithm is derived in :footcite:t:`Byrne1975`. This
        particular implementation follows that implemented in the Matlab routine ode15s
        described in :footcite:t:`shamphine1997matlab` and the SciPy implementation
        :footcite:t:`Virtanen2020`, which features the NDF formulas for improved
        stability with associated differences in the error constants, and calculates
        the jacobian at J(t_{n+1}, y^0_{n+1}). This implementation was based on that
        implemented in the SciPy library :footcite:t:`Virtanen2020`, which also mainly
        follows :footcite:t:`shamphine1997matlab` but uses the more standard Jacobian
        update.

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

        """

        def fun_bind_inputs(y, t):
            return fun(y, t, *args)

        jac_bind_inputs = jax.jacfwd(fun_bind_inputs, argnums=0)

        t0, h0 = t_eval[0], t_eval[1] - t_eval[0]
        stepper = _bdf_init(
            fun_bind_inputs, jac_bind_inputs, mass, t0, y0, h0, rtol, atol
        )
        y_out = jnp.empty((len(t_eval), len(y0)), dtype=y0.dtype)

        def cond_fun(state):
            _, _, i, _ = state
            return i < len(t_eval)

        def body_fun(state):
            stepper, t_eval, i, y_out = state
            stepper = _bdf_step(stepper, fun_bind_inputs, jac_bind_inputs)
            index = jnp.searchsorted(t_eval, stepper.t).astype(jnp.int32)

            def interpolate_and_update(j, y_out):
                y = _bdf_interpolate(stepper, t_eval[j])
                return y_out.at[j].set(y)

            y_out = jax.lax.fori_loop(i, index, interpolate_and_update, y_out)
            return stepper, t_eval, index, y_out

        init_state = (stepper, t_eval, 0, y_out)
        _, _, _, y_out = jax.lax.while_loop(cond_fun, body_fun, init_state)

        return y_out

    BDFState = collections.namedtuple(
        "BDFState",
        [
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
        ],
    )

    jax.tree_util.register_pytree_node(
        BDFState, lambda xs: (tuple(xs), None), lambda _, xs: BDFState(*xs)
    )

    def _bdf_init(fun, jac, mass, t0, y0, h0, rtol, atol):
        """
        Initiation routine for Backward Difference formula (BDF) implicit multistep
        integrator.

        See _bdf_odeint function above for details, this function returns a dict with
        the initial state of the solver

        Parameters
        ----------

        fun: callable
            function with signature (y, t), where t is a scalar time and y is a ndarray
            with shape (n,), returns the rhs of the system of ODE equations as an nd
            array with shape (n,)
        jac: callable
            function with signature (y, t), where t is a scalar time and y is a ndarray
            with shape (n,), returns the jacobian matrix of fun as an ndarray with
            shape (n,n)
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

        EPS = jnp.finfo(y0.dtype).eps

        # Scaling and tolerance initialisation
        scale_y0 = atol + rtol * jnp.abs(y0)
        newton_tol = jnp.maximum(10 * EPS / rtol, jnp.minimum(0.03, rtol**0.5))

        y0, not_converged = _select_initial_conditions(
            fun, mass, t0, y0, newton_tol, scale_y0
        )

        # Compute initial function and step size
        f0 = fun(y0, t0)
        h = _select_initial_step(atol, rtol, fun, t0, y0, f0, h0)

        # Initialise the difference matrix, D
        D = jnp.empty((MAX_ORDER + 1, len(y0)), dtype=y0.dtype)
        D = D.at[jnp.index_exp[0, :]].set(y0)
        D = D.at[jnp.index_exp[1, :]].set(f0 * h)

        # kappa values for difference orders, taken from Table 1 of [1]
        kappa = jnp.array([0, -0.1850, -1 / 9, -0.0823, -0.0415, 0])
        gamma = jnp.hstack((0, jnp.cumsum(1 / jnp.arange(1, MAX_ORDER + 1))))
        alpha = 1.0 / ((1 - kappa) * gamma)
        c = h * alpha[1]
        error_const = kappa * gamma + 1 / jnp.arange(1, MAX_ORDER + 2)

        # Jacobian and LU decomp
        J = jac(y0, t0)
        LU = jax.scipy.linalg.lu_factor(mass - c * J)
        U = _compute_R(1, 1)  # Order 1

        # Create initial BDFState
        state = BDFState(
            t=t0,
            atol=atol,
            rtol=rtol,
            M=mass,
            newton_tol=newton_tol,
            consistent_y0_failed=not_converged,
            order=1,
            h=h,
            n_equal_steps=0,
            D=D,
            y0=y0,
            scale_y0=scale_y0,
            kappa=kappa,
            gamma=gamma,
            alpha=alpha,
            c=c,
            error_const=error_const,
            J=J,
            LU=LU,
            U=U,
            psi=None,
            n_function_evals=2,
            n_jacobian_evals=1,
            n_lu_decompositions=1,
            n_steps=0,
        )

        # Predict initial y0, scale_yo, update state
        y0, scale_y0 = _predict(state, D)
        psi = _update_psi(state, D)
        return state._replace(y0=y0, scale_y0=scale_y0, psi=psi)

    def _compute_R(order, factor):
        """
        computes the R matrix with entries
        given by the first equation on page 8 of [1]

        This is used to update the differences matrix when step size h is varied
        according to factor = h_{n+1} / h_n

        Note that the U matrix also defined in the same section can be also be
        found using factor = 1, which corresponds to R with a constant step size
        """
        I = jnp.arange(1, MAX_ORDER + 1).reshape(-1, 1)
        J = jnp.arange(1, MAX_ORDER + 1)
        M = jnp.empty((MAX_ORDER + 1, MAX_ORDER + 1))
        M = M.at[jnp.index_exp[1:, 1:]].set((I - 1 - factor * J) / I)
        M = M.at[jnp.index_exp[0]].set(1)
        R = jnp.cumprod(M, axis=0)

        return R

    def _select_initial_conditions(fun, M, t0, y0, tol, scale_y0):
        # identify algebraic variables as zeros on diagonal
        algebraic_variables = onp.diag(M) == 0.0

        # if all differentiable variables then return y0 (can use normal python if
        # since M is static)
        if not onp.any(algebraic_variables):
            return y0, False

        # calculate consistent initial conditions via a newton on -J_a @ delta = f_a
        # This follows this reference:
        #
        # Shampine, L. F., Reichelt, M. W., & Kierzenka, J. A. (1999).
        # Solving index-1 DAEs in MATLAB and Simulink. SIAM review, 41(3), 538-552.

        # calculate fun_a, function of algebraic variables
        def fun_a(y_a):
            y_full = y0.at[algebraic_variables].set(y_a)
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
        y_tilde = y0.at[algebraic_variables].set(y_a)

        return y_tilde, converged

    def _select_initial_step(atol, rtol, fun, t0, y0, f0, h0):
        """
        Select a good initial step by stepping forward one step of forward euler, and
        comparing the predicted state against that using the provided function.

        Optimal step size based on the selected order is obtained using formula (4.12)
        in :footcite:t:`Hairer1993`.

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
        orders = jnp.arange(MAX_ORDER + 1)[:, None]
        subD = jnp.where(orders <= state.order, D, 0)
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
        Update of difference equations can be done efficiently
        by reusing d and D.

        From first equation on page 4 of [1]:
        d = y_n - y^0_n = D^{k + 1} y_n

        Standard backwards difference gives
        D^{j + 1} y_n = D^{j} y_n - D^{j} y_{n - 1}

        Combining these gives the following algorithm
        """
        order = state.order
        D = state.D.at[order + 2].set(d - state.D[order + 1])
        D = D.at[order + 1].set(d)

        def update_D(i, D):
            return D.at[order - i].add(D[order - i + 1])

        return jax.lax.fori_loop(0, order + 1, update_D, D)

    def _update_step_size_and_lu(state, factor):
        """
        Update step size and recompute LU decomposition.
        """
        state = _update_step_size(state, factor)
        LU = jax.scipy.linalg.lu_factor(state.M - state.c * state.J)
        return state._replace(LU=LU, n_lu_decompositions=state.n_lu_decompositions + 1)

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
        c = h * state.alpha[order]

        # update D using equations in section 3.2 of [1]
        RU = _compute_R(order, factor).dot(state.U)
        I = jnp.arange(0, MAX_ORDER + 1).reshape(-1, 1)
        J = jnp.arange(0, MAX_ORDER + 1)

        # only update order+1, order+1 entries of D
        RU = jnp.where(
            jnp.logical_and(I <= order, J <= order), RU, jnp.identity(MAX_ORDER + 1)
        )
        D = jnp.dot(RU.T, state.D)

        # update psi, y0 (D has changed)
        psi = _update_psi(state, D)
        y0, scale_y0 = _predict(state, D)

        return state._replace(
            n_equal_steps=0,
            h=h,
            c=c,
            D=D,
            psi=psi,
            y0=y0,
            scale_y0=scale_y0,
        )

    def _update_jacobian(state, jac):
        """
        Update the jacobian using J(t_{n+1}, y^0_{n+1})
        following the scipy bdf implementation rather than J(t_n, y_n) as per [1]
        """
        J = jac(state.y0, state.t + state.h)
        LU = jax.scipy.linalg.lu_factor(state.M - state.c * J)
        return state._replace(
            J=J,
            n_jacobian_evals=state.n_jacobian_evals + 1,
            LU=LU,
            n_lu_decompositions=state.n_lu_decompositions + 1,
        )

    def _newton_iteration(state, fun):
        """
        Perform Newton iteration to solve the system.
        """
        y0 = state.y0
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
            b = state.c * f_eval - state.M @ (state.psi + d)
            dy = jax.scipy.linalg.lu_solve(state.LU, b)
            dy_norm = jnp.sqrt(jnp.mean((dy / scale_y0) ** 2))
            rate = dy_norm / dy_norm_old

            # if iteration is not going to converge in NEWTON_MAXITER
            # (assuming the current rate), then abort
            pred = rate >= 1
            pred += (
                rate ** (NEWTON_MAXITER - k) / (1 - rate) * dy_norm > state.newton_tol
            )
            pred *= dy_norm_old >= 0
            k += pred * (NEWTON_MAXITER - k - 1)

            d += dy
            y = y0 + d

            # if converged then break out of iteration early
            pred = dy_norm_old >= 0.0
            pred *= rate / (1 - rate) * dy_norm < state.newton_tol
            converged = (dy_norm == 0.0) + pred

            dy_norm_old = dy_norm

            return [k + 1, converged, dy_norm_old, d, y, n_function_evals]

        k, converged, dy_norm_old, d, y, n_function_evals = jax.lax.while_loop(
            while_cond, while_body, while_state
        )
        return converged, k, y, d, state._replace(n_function_evals=n_function_evals)

    def rms_norm(arg):
        return jnp.sqrt(jnp.mean(arg**2))

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
            order > 1,
            rms_norm(state.error_const[order - 1] * D[order] / scale_y),
            jnp.inf,
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

        # New step size factor
        factor = jnp.minimum(MAX_FACTOR, safety * factors[max_index])
        new_state = state._replace(D=D, order=order)
        return _update_step_size_and_lu(new_state, factor)

    def _bdf_step(state, fun, jac):
        """
        Perform a BDF step.

        We will try and use the old jacobian unless
        convergence of newton iteration fails.
        """

        def step_iteration(while_state):
            state, updated_jacobian = while_state

            # Solve BDF equation using Newton iteration
            converged, n_iter, y, d, state = _newton_iteration(state, fun)

            # Update Jacobian or reduce step size if not converged
            # Evaluated so reduce step size by 0.3 (as per [1]) and try again
            state, updated_jacobian = jax.lax.cond(
                ~converged,
                lambda s, uj: jax.lax.cond(
                    uj,
                    lambda s: (_update_step_size_and_lu(s, 0.3), True),
                    lambda s: (_update_jacobian(s, jac), True),
                    s,
                ),
                lambda s, uj: (s, uj),
                state,
                updated_jacobian,
            )

            safety = 0.9 * (2 * NEWTON_MAXITER + 1) / (2 * NEWTON_MAXITER + n_iter)
            scale_y = state.atol + state.rtol * jnp.abs(y)

            # Calculate error and updated step size factor
            # combine eq 3, 4 and 6 from [1] to obtain error
            # Note that error = C_k * h^{k+1} y^{k+1}
            # and d = D^{k+1} y_{n+1} \approx h^{k+1} y^{k+1}
            error = state.error_const[state.order] * d
            error_norm = rms_norm(error / scale_y)

            # Calculate optimal step size factor as per eq 2.46 of [2]
            factor = jnp.clip(
                safety * error_norm ** (-1 / (state.order + 1)), MIN_FACTOR, None
            )

            # Update step size if error is too large
            state = jax.lax.cond(
                converged & (error_norm > 1),
                lambda s: _update_step_size_and_lu(s, factor),
                lambda s: s,
                state,
            )

            step_accepted = converged & (error_norm <= 1)
            return (state, updated_jacobian), (step_accepted, y, d, n_iter)

        # Iterate until step is accepted
        (state, _), (_, y, d, n_iter) = jax.lax.while_loop(
            lambda carry_and_aux: ~carry_and_aux[1][0],
            lambda carry_and_aux: step_iteration(carry_and_aux[0]),
            (
                (state, False),
                (False, jnp.empty_like(state.y0), jnp.empty_like(state.y0), -1),
            ),
        )

        # Update state for the accepted step
        n_steps = state.n_steps + 1
        t = state.t + state.h
        n_equal_steps = state.n_equal_steps + 1
        state = state._replace(n_equal_steps=n_equal_steps, t=t, n_steps=n_steps)

        # Prepare for the next step, potentially changing order
        # (see page 83 of [2])
        state = jax.lax.cond(
            n_equal_steps < state.order + 1,
            lambda s: _prepare_next_step(s, d),
            lambda s: _prepare_next_step_order_change(s, d, y, n_iter),
            state,
        )

        return state

    def _bdf_interpolate(state, t_eval):
        """
        interpolate solution at time values t* where t-h < t* < t

        definition of the interpolating polynomial can be found on page 7 of [1]
        """
        h = state.h
        D = state.D
        j = 0
        time_factor = 1.0
        order_summation = D[0]
        while_state = [j, time_factor, order_summation]

        def while_cond(while_state):
            j, _, _ = while_state
            return j < state.order

        def while_body(while_state):
            j, time_factor, order_summation = while_state
            time_factor *= (t_eval - (state.t - h * j)) / (h * (1 + j))
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
    # edits), has been modified from the JAX library at https://github.com/jax-ml/jax.
    # The main difference is the addition of support for semi-explicit dae index 1
    # problems via the addition of a mass matrix.
    # This is under an Apache license, a short form of which is given here:
    #
    # Copyright 2018 Google LLC
    #
    # Licensed under the Apache License, Version 2.0 (the "License"); you may not use
    # this file except in compliance with the License.
    # You may obtain a copy of the License at
    #
    #     https://www.apache.org/licenses/LICENSE-2.0
    #
    # Unless required by applicable law or agreed to in writing, software distributed
    # under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
    # CONDITIONS OF ANY KIND, either express or implied.  See the License for the
    # specific language governing permissions and limitations under the License.

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

    @partial(gnool_jit, static_array_argnums=(0, 1, 2, 3))
    def _bdf_odeint_wrapper(func, mass, rtol, atol, y0, ts, *args):
        y0, unravel = ravel_pytree(y0)
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
            # analysis for differential-algebraic equations: The adjoint DAE system and
            # its numerical solution.
            # SIAM journal on scientific computing, 24(3), 1076-1089.
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
                y_bar = g0.at[differentiable_variables].set(
                    jax.scipy.linalg.lu_solve(LU_invM_dd, g0_a - J_ad @ invJ_aa)
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
            Note:JAX puts in empty arrays into args for some reason, we remove them here
            """
            return sum((tuple(b.values()) for b in args if isinstance(b, dict)), ())

        aug_mass = (
            mass,
            mass,
            onp.array(1.0),
            *arg_dicts_to_values(tree_map(arg_to_identity, args)),
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
            y_bar, t0_bar, args_bar = tree_map(
                op.itemgetter(1), (y_bar, t0_bar, args_bar)
            )
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
        wrapped_fun, out_tree = flatten_fun_nokwargs(lu.wrap_init(fun), in_tree)
        jaxpr, _, consts = pe.trace_to_jaxpr_dynamic(wrapped_fun, in_avals)
        out_tree = out_tree()

        # We only want to closure convert for constants with respect to which we're
        # differentiating. As a proxy for that, we hoist consts with float dtype.
        # TODO(mattjj): revise this approach
        def is_float(c):
            return dtypes.issubdtype(type(c), jnp.inexact)

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
        ans = yield (y, *args), {}
        ans_flat, _ = ravel_pytree(ans)
        yield ans_flat


def jax_bdf_integrate(func, y0, t_eval, *args, rtol=1e-6, atol=1e-6, mass=None):
    """
    Backward Difference formula (BDF) implicit multistep integrator. The basic algorithm
    is derived in :footcite:t:`Byrne1975`. This particular implementation
    follows that implemented in the Matlab routine ode15s described in
    :footcite:t:`Shampine1997` and the SciPy implementation
    :footcite:t:`Virtanen2020` which features the NDF formulas for improved stability,
    with associated differences in the error constants, and calculates the jacobian at
    J(t_{n+1}, y^0_{n+1}). This implementation was based on that implemented in the
    SciPy library :footcite:t:`Virtanen2020`, which also mainly follows
    :footcite:t:`Shampine1997` but uses the more standard jacobian update.

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

    """
    if not pybamm.has_jax():
        raise ModuleNotFoundError(
            "Jax or jaxlib is not installed, please see https://docs.pybamm.org/en/latest/source/user_guide/installation/gnu-linux-mac.html#optional-jaxsolver"
        )

    def _check_arg(arg):
        if not isinstance(arg, core.Tracer) and not core.valid_jaxtype(arg):
            msg = (
                "The contents of odeint *args must be arrays or scalars, but got \n{}."
            )
        raise TypeError(msg.format(arg))

    flat_args, in_tree = tree_flatten((y0, t_eval[0], *args))
    in_avals = tuple(safe_map(abstractify, flat_args))
    converted, consts = closure_convert(func, in_tree, in_avals)
    if mass is None:
        mass = onp.identity(y0.shape[0], dtype=y0.dtype)
    else:
        mass = block_diag(tree_flatten(mass)[0])
    return _bdf_odeint_wrapper(converted, mass, rtol, atol, y0, t_eval, *consts, *args)
