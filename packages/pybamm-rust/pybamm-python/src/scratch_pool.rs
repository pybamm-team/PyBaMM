use std::sync::Mutex;

/// A zeroed buffer of `len` slots, the scratch one `CompiledExpr` evaluation needs.
pub fn buffer(len: usize) -> Box<[f64]> {
    vec![0.0; len].into_boxed_slice()
}

/// Reuse pool for the per-call scratch of a frozen class.
///
/// `PyO3` methods take `&self` and may run concurrently once the GIL is released, so
/// each call needs scratch of its own. A call takes an item from the pool, or makes one
/// when the pool is empty or locked by another call, and returns it afterwards, so a
/// steady-state call does not allocate.
#[derive(Debug)]
pub struct ScratchPool<S> {
    items: Mutex<Vec<S>>,
}

impl<S> Default for ScratchPool<S> {
    fn default() -> Self {
        Self {
            items: Mutex::new(Vec::new()),
        }
    }
}

impl<S> ScratchPool<S> {
    /// Retained-item cap: bounds steady-state memory under thread churn.
    pub const MAX_POOLED: usize = 8;

    /// A pooled item, else `make()`. A pooled item may carry stale data, so callers must
    /// overwrite every slot before reading.
    pub fn acquire(&self, make: impl FnOnce() -> S) -> S {
        if let Ok(mut pool) = self.items.try_lock()
            && let Some(item) = pool.pop()
        {
            return item;
        }
        make()
    }

    pub fn release(&self, item: S) {
        if let Ok(mut pool) = self.items.try_lock()
            && pool.len() < Self::MAX_POOLED
        {
            pool.push(item);
        }
    }

    #[cfg(test)]
    pub fn pooled_len(&self) -> usize {
        self.items.lock().map_or(0, |p| p.len())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn acquire_release_reuses_buffer() {
        let pool = ScratchPool::default();
        let a = pool.acquire(|| buffer(16));
        assert_eq!(a.len(), 16);
        let ptr = a.as_ptr();
        pool.release(a);
        let b = pool.acquire(|| unreachable!("a pooled buffer is available"));
        // Ownership round-trips, so pointer identity is not allocator-dependent.
        assert_eq!(b.as_ptr(), ptr, "steady state must reuse the buffer");
    }

    #[test]
    fn pool_caps_retained_buffers() {
        let pool = ScratchPool::default();
        let bufs: Vec<_> = (0..(ScratchPool::<Box<[f64]>>::MAX_POOLED + 4))
            .map(|_| pool.acquire(|| buffer(4)))
            .collect();
        for b in bufs {
            pool.release(b);
        }
        assert!(pool.pooled_len() <= ScratchPool::<Box<[f64]>>::MAX_POOLED);
    }
}
