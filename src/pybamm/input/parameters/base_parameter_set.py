class AbstractBaseParameters:
    _details = {}
    _sei = {}
    _cell = {}
    _negative_electrode = {}
    _positive_electrode = {}
    _seperator = {}
    _electrolyte = {}
    _experiment = {}

    def get_param_set(self):
        full_set = {}
        for sub_set in [
            self._details,
            self._sei,
            self._cell,
            self._negative_electrode,
            self._positive_electrode,
            self._seperator,
            self._electrolyte,
            self._experiment,
        ]:
            full_set.update(sub_set)
        return full_set
