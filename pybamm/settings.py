#
# Settings class for PyBaMM
#


class Settings(object):
    _debug_mode = False
    min_max_heaviside_smoothing_parameter = "exact"

    @property
    def debug_mode(self):
        return self._debug_mode

    @debug_mode.setter
    def debug_mode(self, value):
        assert isinstance(value, bool)
        self._debug_mode = value


settings = Settings()
