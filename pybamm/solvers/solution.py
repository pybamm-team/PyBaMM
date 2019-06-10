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
    termination : str
        String to indicate why the solution terminated

    """

    def __init__(self, t, y, termination):
        self.t = t
        self.y = y
        self.termination = termination

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

    @property
    def termination(self):
        "Reason for termination"
        return self._termination

    @termination.setter
    def termination(self, value):
        "Updates the reason for termination"
        self._termination = value
