#
# Solution class
#


class Solution(object):
    """
    Class containing the solution of, and various attributes associated with, a PyBaMM
    model.
    """

    def __init__(self, t, y):
        self.t = t
        self.y = y

    @property
    def t(self):
        return self._t

    @t.setter
    def t(self, value):
        self._t = value

    @property
    def y(self):
        return self._y

    @y.setter
    def y(self, value):
        self._y = value
