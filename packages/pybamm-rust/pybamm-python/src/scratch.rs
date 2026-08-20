use std::sync::Mutex;

use pybamm_core::{JacobianData, JacobianScratch};

/// How a [`ScratchPool`] makes a buffer when it has none to hand out.
///
/// The recipe rather than a spare buffer: a pool that minted by cloning a
/// prototype would both retain one extra set of buffers for its lifetime and
/// memcpy them on every miss, where the shape alone is what a caller needs.
pub trait Mint {
    /// What this recipe produces.
    type Item;

    /// A fresh item, with every slot zeroed.
    fn mint(&self) -> Self::Item;
}

/// Scratch for one `CompiledExpr`, named by the buffer length it needs.
#[derive(Debug)]
pub struct Buffer(pub usize);

impl Mint for Buffer {
    type Item = Box<[f64]>;

    fn mint(&self) -> Self::Item {
        vec![0.0; self.0].into_boxed_slice()
    }
}

impl Mint for std::sync::Arc<JacobianData> {
    type Item = JacobianScratch;

    fn mint(&self) -> Self::Item {
        JacobianScratch::new(self)
    }
}

/// Reuse pool for the per-call scratch an evaluation needs.
///
/// `PyO3` methods take `&self`, so scratch cannot live in the object; pooling it
/// keeps a steady-state call from allocating. Concurrent calls that find the pool
/// locked mint their own rather than block.
#[derive(Debug)]
pub struct ScratchPool<M: Mint> {
    buffers: Mutex<Vec<M::Item>>,
    mint: M,
}

impl<M: Mint> ScratchPool<M> {
    /// Retained-buffer cap: bounds steady-state memory under thread churn.
    pub const MAX_POOLED: usize = 8;

    pub const fn new(mint: M) -> Self {
        Self {
            buffers: Mutex::new(Vec::new()),
            mint,
        }
    }

    /// Pooled buffers may carry stale data from a prior evaluation;
    /// callers must overwrite every slot before reading (the tape in
    /// `CompiledExpr::eval` satisfies this — every slot is written
    /// before any downstream read).
    pub fn acquire(&self) -> M::Item {
        if let Ok(mut pool) = self.buffers.try_lock()
            && let Some(buf) = pool.pop()
        {
            return buf;
        }
        self.mint.mint()
    }

    pub fn release(&self, buf: M::Item) {
        if let Ok(mut pool) = self.buffers.try_lock()
            && pool.len() < Self::MAX_POOLED
        {
            pool.push(buf);
        }
    }

    #[cfg(test)]
    pub fn pooled_len(&self) -> usize {
        self.buffers.lock().map_or(0, |p| p.len())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn acquire_release_reuses_buffer() {
        let pool = ScratchPool::new(Buffer(16));
        let a = pool.acquire();
        assert_eq!(a.len(), 16);
        let ptr = a.as_ptr();
        pool.release(a);
        let b = pool.acquire();
        // ownership moves into the pool and back out — same allocation,
        // so pointer identity is guaranteed (not allocator-dependent)
        assert_eq!(b.as_ptr(), ptr, "steady state must reuse the buffer");
    }

    #[test]
    fn pool_caps_retained_buffers() {
        let pool = ScratchPool::new(Buffer(4));
        let bufs: Vec<_> = (0..(ScratchPool::<Buffer>::MAX_POOLED + 4))
            .map(|_| pool.acquire())
            .collect();
        for b in bufs {
            pool.release(b);
        }
        assert!(pool.pooled_len() <= ScratchPool::<Buffer>::MAX_POOLED);
    }
}
