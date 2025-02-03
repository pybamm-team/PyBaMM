import types


class NaturalNumberOption:
    def __init__(self, default_value):
        self.value = default_value

    def __contains__(self, value):
        is_an_integer = isinstance(value, int)
        is_non_negative = value >= 0
        return is_an_integer and is_non_negative

    def __getitem__(self, value):
        return self.value

    def __repr__(self):
        return "natural numbers (e.g. 0, 1, 2, 3, ...)"


class OperatingModes:
    def __init__(self, default_mode):
        self.default_mode = default_mode

        self.named_modes = [
            "current",
            "voltage",
            "power",
            "differential power",
            "explicit power",
            "resistance",
            "differential resistance",
            "explicit resistance",
            "CCCV",
        ]

    def __contains__(self, value):
        named_mode = value in self.named_modes
        function = isinstance(value, types.FunctionType)
        return named_mode or function

    def __getitem__(self, value):
        return self.default_mode
