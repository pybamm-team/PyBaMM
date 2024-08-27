class AbstractBaseParameters:
    """
    Base class for grouping and processing parameter sets.
    """

    def __init__(self):
        self._details = {}
        self._plating = {}
        self._sei = {}
        self._thermal = {}
        self._cell = {}
        self._negative_electrode = {}
        self._positive_electrode = {}
        self._separator = {}
        self._electrolyte = {}

    def degradation_available(self):
        return True if self._sei else False

    def thermal_available(self):
        return True if self._thermal else False

    def plating_available(self):
        return True if self._plating else False

    def get_param_set(self):
        full_set = {}
        for sub_set in [
            self._details,
            self._plating,
            self._sei,
            self._thermal,
            self._cell,
            self._negative_electrode,
            self._positive_electrode,
            self._separator,
            self._electrolyte,
        ]:
            full_set.update(sub_set)
        return full_set
