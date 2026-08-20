"""The native diffsol solver must release the GIL during integration.

A pure-Rust call that holds the GIL freezes every other Python thread for its
entire duration (there are no bytecode boundaries, so the periodic GIL switch
never fires). With ``py.detach`` around the integration another thread keeps
getting scheduled throughout the solve. This test guards that release so a
future change that drops the ``detach`` is caught.
"""

import threading
import time

import numpy as np

from pybamm.rust import CompiledModel, ExprGraph, PreparedSolver


def _decay_prepared_solver(n_states):
    """Native prepared solver for a decoupled linear decay system dy_i/dt=-y_i."""
    g = ExprGraph()
    terms = [g.mul(g.scalar(-1.0), g.state_vector(i, i + 1)) for i in range(n_states)]
    rhs = g.concat(terms)
    model = CompiledModel.from_expr(
        g,
        rhs,
        np.ones(n_states),  # identity mass matrix (CSR)
        np.arange(n_states + 1, dtype=np.int64),
        np.arange(n_states, dtype=np.int64),
        n_inputs=0,
        sens_param_indices=[],
        output_exprs=[],
        event_exprs=[],
    )
    return PreparedSolver(model, 1e-8, 1e-8)


def test_native_solve_releases_gil():
    n_states = 100
    ps = _decay_prepared_solver(n_states)
    y0 = np.ones(n_states)
    inputs = np.array([], dtype=np.float64)
    no_stops = np.array([], dtype=np.float64)
    # A fine output grid makes one native solve take hundreds of milliseconds.
    t_eval = np.linspace(0.0, 5.0, 400_000)

    ps.solve(
        t_eval, no_stops, y0, inputs
    )  # warm up (first solve builds internal state)

    stop = threading.Event()
    ticks = []

    def worker():
        # Each tick needs the GIL only briefly, and sleeping keeps this thread off
        # the CPU so it never competes with the solve for cores or memory.
        while not stop.is_set():
            ticks.append(time.perf_counter())
            time.sleep(0.001)

    t = threading.Thread(target=worker)
    t.start()
    try:
        # The cadence the worker keeps with the main thread idle. Sleep granularity
        # varies by platform, so this is the yardstick, not a wall-clock constant.
        time.sleep(0.1)
        quiet_cadence = np.percentile(np.diff(ticks[:]), 95)

        wall0 = time.perf_counter()
        ps.solve(t_eval, no_stops, y0, inputs)
        wall1 = time.perf_counter()
    finally:
        stop.set()
        t.join()

    solve_wall = wall1 - wall0
    assert solve_wall > 8 * quiet_cadence, (
        f"native solve took {solve_wall * 1e3:.1f} ms against a worker cadence of "
        f"{quiet_cadence * 1e3:.1f} ms, too few ticks to judge the GIL either way; "
        "raise the output-point count"
    )

    # A held GIL blocks the worker outright, so it records no tick inside the call.
    during = [tick for tick in ticks if wall0 < tick < wall1]
    assert during, (
        f"worker ran zero times during a {solve_wall * 1e3:.1f} ms native solve: "
        "the GIL was not released"
    )

    # A GIL held for part of the solve leaves one long quiet stretch, which is what
    # to bound; the worker's throughput would also track CPU and memory contention.
    longest_gap = np.diff([wall0, *during, wall1]).max()
    assert longest_gap < 0.5 * solve_wall, (
        f"worker stalled for {longest_gap * 1e3:.1f} ms inside a "
        f"{solve_wall * 1e3:.1f} ms native solve: the GIL was held for part of it"
    )
