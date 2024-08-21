class AbstractBaseParameters:
    _details = {}
    _sei = {}
    _thermal = {}
    _cell = {}
    _negative_electrode = {}
    _positive_electrode = {}
    _seperator = {}
    _electrolyte = {}

    def degradation_available(self):
        return not self._sei

    def thermal_available(self):
        return not self._thermal

    def get_param_set(self):
        full_set = {}
        for sub_set in [
            self._details,
            self._sei,
            self._thermal,
            self._cell,
            self._negative_electrode,
            self._positive_electrode,
            self._seperator,
            self._electrolyte,
        ]:
            full_set.update(sub_set)
        return full_set
