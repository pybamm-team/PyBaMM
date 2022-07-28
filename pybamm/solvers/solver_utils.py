#
# Utility functions and classes for solvers
#


class NoMemAllocVertcat:
    """
    Acts like a vertcat, but does not allocate new memory.
    """

    def __init__(self, a, b):
        arrays = [a, b]
        self.arrays = arrays

        for array in arrays:
            if not 1 <= len(array.shape) <= 2:
                raise ValueError("Only 1D or 2D arrays are supported")
            self._ndim = len(array.shape)

        self.len_a = a.shape[0]
        shape0 = a.shape[0] + b.shape[0]

        if self._ndim == 1:
            self._shape = (shape0,)
            self._size = shape0
        else:
            if a.shape[1] != b.shape[1]:
                raise ValueError("All arrays must have the same number of columns")
            shape1 = a.shape[1]

            self._shape = (shape0, shape1)
            self._size = shape0 * shape1

    @property
    def shape(self):
        return self._shape

    @property
    def size(self):
        return self._size

    @property
    def ndim(self):
        return self._ndim

    def __getitem__(self, key):
        if self._ndim == 1 or isinstance(key, int):
            if key < self.len_a:
                return self.arrays[0][key]
            else:
                return self.arrays[1][key - self.len_a]

        if key[0] == slice(None):
            return NoMemAllocVertcat(*[arr[:, key[1]] for arr in self.arrays])
        elif isinstance(key[0], int):
            if key[0] < self.len_a:
                return self.arrays[0][key[0], key[1]]
            else:
                return self.arrays[1][key[0] - self.len_a, key[1]]
        else:
            raise NotImplementedError
