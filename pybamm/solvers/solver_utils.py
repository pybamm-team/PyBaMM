#
# Utility functions and classes for solvers
#
import casadi


class NoMemAllocVertcat:
    """
    Acts like a vertcat, but does not allocate new memory.
    """

    def __init__(self, xs, ys, len_x=None, len_y=None, items=None):
        self.xs = xs
        self.ys = ys
        self.len_x = len_x or xs[0].shape[0]
        self.len_y = len_y or ys[0].shape[0]
        len_items = len(xs)
        self.shape = (self.len_x + self.len_y, len_items)

        if items is None:
            items = [None] * len_items
        self.items = items

    def __getitem__(self, idx):
        if idx[0] != slice(None):
            raise NotImplementedError(
                "Only full slices are supported in the first entry of the index"
            )
        idx = idx[1]
        if isinstance(idx, slice):
            return NoMemAllocVertcat(
                self.xs[idx], self.ys[idx], self.len_x, self.len_y, self.items[idx]
            )
        else:
            item = self.items[idx]
            if item is not None:
                return item
            else:
                out = casadi.DM.zeros((self.shape[0], 1))
                out[: self.len_x] = self.xs[idx]
                out[self.len_x :] = self.ys[idx]
                self.items[idx] = out
                return out
