#
# Utility functions and classes for solvers
#
import casadi


class NoMemAllocVertcat:
    """
    Acts like a vertcat, but does not allocate new memory.
    """

    def __init__(self, xs, zs, len_x=None, len_z=None, items=None):
        self.xs = xs
        self.zs = zs
        self.len_x = len_x or xs[0].shape[0]
        self.len_z = len_z or zs[0].shape[0]
        len_items = len(xs)
        self.shape = (self.len_x + self.len_z, len_items)

        if items is None:
            items = [None] * len_items
            for idx in range(len_items):
                out = casadi.DM.zeros((self.shape[0], 1))
                out[: self.len_x] = self.xs[idx]
                out[self.len_x :] = self.zs[idx]
                items[idx] = out

        self.items = items

    def __getitem__(self, idx):
        if idx[0] != slice(None):
            raise NotImplementedError(
                "Only full slices are supported in the first entry of the index"
            )
        idx = idx[1]
        if isinstance(idx, slice):
            return NoMemAllocVertcat(
                self.xs[idx], self.zs[idx], self.len_x, self.len_z, self.items[idx]
            )
        else:
            return self.items[idx]
