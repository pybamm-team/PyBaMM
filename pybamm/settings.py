#
# Settings class for PyBaMM
#


class Settings(object):
    _debug_mode = False
    _simplify = True
    _min_smoothing = "exact"
    _max_smoothing = "exact"
    _heaviside_smoothing = "exact"
    _abs_smoothing = "exact"
    max_words_in_line = 4

    @property
    def debug_mode(self):
        return self._debug_mode

    @debug_mode.setter
    def debug_mode(self, value):
        assert isinstance(value, bool)
        self._debug_mode = value

    @property
    def simplify(self):
        return self._simplify

    @simplify.setter
    def simplify(self, value):
        assert isinstance(value, bool)
        self._simplify = value

    def set_smoothing_parameters(self, k):
        "Helper function to set all smoothing parameters"
        self.min_smoothing = k
        self.max_smoothing = k
        self.heaviside_smoothing = k
        self.abs_smoothing = k

    @property
    def min_smoothing(self):
        return self._min_smoothing

    @min_smoothing.setter
    def min_smoothing(self, k):
        if k != "exact" and k <= 0:
            raise ValueError(
                "smoothing parameter must be 'exact' or a strictly positive number"
            )
        self._min_smoothing = k

    @property
    def max_smoothing(self):
        return self._max_smoothing

    @max_smoothing.setter
    def max_smoothing(self, k):
        if k != "exact" and k <= 0:
            raise ValueError(
                "smoothing parameter must be 'exact' or a strictly positive number"
            )
        self._max_smoothing = k

    @property
    def heaviside_smoothing(self):
        return self._heaviside_smoothing

    @heaviside_smoothing.setter
    def heaviside_smoothing(self, k):
        if k != "exact" and k <= 0:
            raise ValueError(
                "smoothing parameter must be 'exact' or a strictly positive number"
            )
        self._heaviside_smoothing = k

    @property
    def abs_smoothing(self):
        return self._abs_smoothing

    @abs_smoothing.setter
    def abs_smoothing(self, k):
        if k != "exact" and k <= 0:
            raise ValueError(
                "smoothing parameter must be 'exact' or a strictly positive number"
            )
        self._abs_smoothing = k


settings = Settings()
