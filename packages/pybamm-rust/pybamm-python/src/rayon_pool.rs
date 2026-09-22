//! Process-wide rayon pools, keyed by thread count and shared by all solvers.

use std::sync::{Arc, LazyLock, Mutex, MutexGuard, PoisonError};

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rayon::{ThreadPool, ThreadPoolBuilder};

/// Pools kept at once; building another drops the least recently used from the cache.
const MAX_CACHED_POOLS: usize = 4;

/// The pools built by process `pid`, least recently used first.
struct PoolCache {
    pid: u32,
    pools: Vec<(usize, Arc<ThreadPool>)>,
}

impl PoolCache {
    const fn new(pid: u32) -> Self {
        Self {
            pid,
            pools: Vec::new(),
        }
    }

    /// The cached pool of `threads` workers, now the most recently used.
    fn get(&mut self, threads: usize) -> Option<Arc<ThreadPool>> {
        let index = self.pools.iter().position(|(n, _)| *n == threads)?;
        let entry = self.pools.remove(index);
        let pool = Arc::clone(&entry.1);
        self.pools.push(entry);
        Some(pool)
    }

    /// Cache `pool` unless a racing build got there first, returning the pool to use and
    /// any pool evicted to make room.
    fn insert(
        &mut self,
        threads: usize,
        pool: Arc<ThreadPool>,
    ) -> (Arc<ThreadPool>, Option<Arc<ThreadPool>>) {
        if let Some(cached) = self.get(threads) {
            return (cached, None);
        }
        let evicted = (self.pools.len() == MAX_CACHED_POOLS).then(|| self.pools.remove(0).1);
        self.pools.push((threads, Arc::clone(&pool)));
        (pool, evicted)
    }
}

static POOLS: LazyLock<Mutex<PoolCache>> =
    LazyLock::new(|| Mutex::new(PoolCache::new(std::process::id())));

/// Lock this process's pool cache, ignoring poisoning (a panic cannot leave it
/// observably corrupt).
fn lock_cache() -> MutexGuard<'static, PoolCache> {
    let mut cache = POOLS.lock().unwrap_or_else(PoisonError::into_inner);
    let pid = std::process::id();
    if cache.pid != pid {
        // A forked child inherits the pools but not their workers, so a job would wait
        // forever. Leak them: dropping one can block on a lock a lost worker held.
        for (_, pool) in cache.pools.drain(..) {
            std::mem::forget(pool);
        }
        cache.pid = pid;
    }
    cache
}

/// A fresh pool of `threads` named workers.
fn build_pool(threads: usize) -> PyResult<ThreadPool> {
    ThreadPoolBuilder::new()
        .num_threads(threads)
        .thread_name(|i| format!("pybamm-solve-{i}"))
        .build()
        .map_err(|e| {
            PyRuntimeError::new_err(format!(
                "could not build a rayon pool of {threads} thread(s): {e}"
            ))
        })
}

/// The shared pool of `threads` workers, built on first request.
pub fn pool_for(threads: usize) -> PyResult<Arc<ThreadPool>> {
    pool_for_with(threads, build_pool)
}

/// [`pool_for`], building a missing pool with `build`.
fn pool_for_with(
    threads: usize,
    build: impl FnOnce(usize) -> PyResult<ThreadPool>,
) -> PyResult<Arc<ThreadPool>> {
    let cached = lock_cache().get(threads);
    if let Some(pool) = cached {
        return Ok(pool);
    }
    // Built unlocked: a fork while another thread holds the lock deadlocks the child.
    let built = Arc::new(build(threads)?);
    let (pool, evicted) = lock_cache().insert(threads, built);
    // A solve still running on an evicted pool keeps it alive until it returns.
    drop(evicted);
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

#[cfg(test)]
mod tests {
    use super::*;

    /// A one-worker pool, standing in for one of any width.
    fn small_pool() -> Arc<ThreadPool> {
        Arc::new(build_pool(1).expect("pool build failed"))
    }

    fn cached_widths(cache: &PoolCache) -> Vec<usize> {
        cache.pools.iter().map(|(threads, _)| *threads).collect()
    }

    #[test]
    fn the_cache_keeps_only_the_most_recently_used_pools() {
        let mut cache = PoolCache::new(std::process::id());
        let first = small_pool();
        assert!(cache.insert(1, Arc::clone(&first)).1.is_none());
        for threads in 2..=MAX_CACHED_POOLS {
            assert!(cache.insert(threads, small_pool()).1.is_none());
        }

        // Reading the oldest makes width 2 the least recently used.
        let read = cache.get(1).expect("width 1 cached");
        assert!(Arc::ptr_eq(&read, &first));
        let newest = MAX_CACHED_POOLS + 1;
        let (_, evicted) = cache.insert(newest, small_pool());
        assert!(evicted.is_some(), "a full cache must evict to grow");
        let widths = cached_widths(&cache);
        assert_eq!(widths.len(), MAX_CACHED_POOLS);
        assert!(!widths.contains(&2), "{widths:?}");
        assert_eq!(widths[MAX_CACHED_POOLS - 2..], [1, newest]);
    }

    #[test]
    fn a_racing_build_hands_back_the_pool_already_cached() {
        let mut cache = PoolCache::new(std::process::id());
        let first = small_pool();
        cache.insert(2, Arc::clone(&first));
        let (pool, evicted) = cache.insert(2, small_pool());
        assert!(Arc::ptr_eq(&pool, &first));
        assert!(evicted.is_none());
        assert_eq!(cached_widths(&cache), [2]);
    }

    #[test]
    fn a_pool_is_built_without_holding_the_cache_lock() {
        // The only test on the process-wide cache, so no other test holds its lock.
        let pool = pool_for_with(3, |threads| {
            assert!(
                POOLS.try_lock().is_ok(),
                "the cache lock is held across the build"
            );
            build_pool(threads)
        })
        .expect("pool");
        assert!(Arc::ptr_eq(&pool, &pool_for(3).expect("cached pool")));
    }
}
