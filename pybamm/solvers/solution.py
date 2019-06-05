#
# Solution class
#


class Solution(object):
    """
    Class containing the solution of, and various attributes associated with, a PyBaMM
    model.

    Parameters
    ----------
    t : :class:`numpy.array`, size (n,)
        A one-dimensional array containing the times at which the solution is evaluated
    y : :class:`numpy.array`, size (m, n)
        A two-dimensional array containing the values of the solution. y[i, :] is the
        vector of solutions at time t[i].

    """

    def __init__(self, t, y):
        self.t = t
        self.y = y

    @property
    def t(self):
        "Times at which the solution is evaluated"
        return self._t

    @t.setter
    def t(self, value):
        "Updates the solution times"
        self._t = value

    @property
    def y(self):
        "Values of the solution"
        return self._y

    @y.setter
    def y(self, value):
        "Updates the solution values"
        self._y = value
