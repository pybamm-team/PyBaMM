//! Process-wide rayon pools, one per distinct thread count.
//!
//! Pool ownership sits in the binding rather than in `pybamm-core`: a numerics
//! library should not own process-wide OS threads, while a binding is the layer
//! that knows "one process, many solver objects". Non-Python consumers of the
//! core install their own pool and lose nothing.
//!
//! One solver per experiment step is the common case, so pools are keyed by
//! count rather than owned per solver: ten `DiffsolSolver`s asking for 8 threads
//! share one pool of 8, not eighty threads. Idle rayon workers park on a condvar
//! after a short spin, so a cached pool costs stack address space and no CPU.
//!
//! `ThreadPool::install` re-entered from inside the same pool runs inline, so
//! nested batch calls cannot deadlock.

use std::collections::HashMap;
use std::sync::{Arc, LazyLock, Mutex, MutexGuard, PoisonError};

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rayon::{ThreadPool, ThreadPoolBuilder};

/// Populated lazily, so a process that never asks for more than one thread
/// never constructs rayon at all.
static POOLS: LazyLock<Mutex<HashMap<usize, Arc<ThreadPool>>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

/// Lock the pool cache, recovering from a poisoned lock.
///
/// A panic inside a locked section can only have left the `HashMap` mid-insert,
/// which no reader can observe as corrupt, so poisoning is not a reason to fail
/// every later solve.
fn lock_cache() -> MutexGuard<'static, HashMap<usize, Arc<ThreadPool>>> {
    POOLS.lock().unwrap_or_else(PoisonError::into_inner)
}

/// The shared pool of `threads` workers, building it on first request.
///
/// Workers are named `pybamm-solve-<i>` so `top -H`, `perf` and py-spy identify
/// them. The build runs under the lock, so two solvers constructed concurrently
/// share one pool rather than racing to create two.
pub fn pool_for(threads: usize) -> PyResult<Arc<ThreadPool>> {
    let mut pools = lock_cache();
    if let Some(pool) = pools.get(&threads) {
        return Ok(Arc::clone(pool));
    }
    let pool = Arc::new(
        ThreadPoolBuilder::new()
            .num_threads(threads)
            .thread_name(|i| format!("pybamm-solve-{i}"))
            .build()
            .map_err(|e| {
                PyRuntimeError::new_err(format!(
                    "could not build a rayon pool of {threads} thread(s): {e}"
                ))
            })?,
    );
    pools.insert(threads, Arc::clone(&pool));
    // Explicit, so the guard's scope stays tight enough for clippy.
    drop(pools);
    Ok(pool)
}

/// The cached pools as `{thread_count: pool identity}`.
///
/// Introspection for the tests that pin the caching rules — that solvers asking
/// for the same width share one pool, and that a default-configured process
/// builds none.
#[pyfunction]
pub fn _pool_ids(py: Python<'_>) -> PyResult<Py<PyDict>> {
    let dict = PyDict::new(py);
    // Integer keys run no arbitrary Python, so set_item cannot re-enter
    // pool_for and the cache stays locked for the whole build.
    for (threads, pool) in lock_cache().iter() {
        dict.set_item(threads, Arc::as_ptr(pool) as usize)?;
    }
    Ok(dict.unbind())
}
