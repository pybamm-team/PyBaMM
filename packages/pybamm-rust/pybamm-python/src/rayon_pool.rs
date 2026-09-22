//! Process-wide rayon pools, one per distinct thread count, shared by all solvers.

use std::collections::HashMap;
use std::sync::{Arc, LazyLock, Mutex, MutexGuard, PoisonError};

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rayon::{ThreadPool, ThreadPoolBuilder};

/// The pools built by process `pid`, keyed by thread count.
struct PoolCache {
    pid: u32,
    pools: HashMap<usize, Arc<ThreadPool>>,
}

static POOLS: LazyLock<Mutex<PoolCache>> = LazyLock::new(|| {
    Mutex::new(PoolCache {
        pid: std::process::id(),
        pools: HashMap::new(),
    })
});

/// Lock this process's pool cache, ignoring poisoning (a panic cannot leave it
/// observably corrupt).
fn lock_cache() -> MutexGuard<'static, PoolCache> {
    let mut cache = POOLS.lock().unwrap_or_else(PoisonError::into_inner);
    let pid = std::process::id();
    if cache.pid != pid {
        // A forked child inherits the pools but not their workers, so a job would wait
        // forever. Leak them: dropping one can block on a lock a lost worker held.
        for (_, pool) in cache.pools.drain() {
            std::mem::forget(pool);
        }
        cache.pid = pid;
    }
    cache
}

/// The shared pool of `threads` workers, built on first request.
pub fn pool_for(threads: usize) -> PyResult<Arc<ThreadPool>> {
    let mut cache = lock_cache();
    if let Some(pool) = cache.pools.get(&threads) {
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
    cache.pools.insert(threads, Arc::clone(&pool));
    // Explicit, to keep the guard's scope tight for clippy.
    drop(cache);
    Ok(pool)
}

/// The cached pools as `{thread_count: pool identity}`, for tests.
#[pyfunction]
pub fn _pool_ids(py: Python<'_>) -> PyResult<Py<PyDict>> {
    let dict = PyDict::new(py);
    // Integer keys run no Python, so `set_item` cannot re-enter `pool_for` under the lock.
    for (threads, pool) in &lock_cache().pools {
        dict.set_item(threads, Arc::as_ptr(pool).addr())?;
    }
    Ok(dict.unbind())
}
