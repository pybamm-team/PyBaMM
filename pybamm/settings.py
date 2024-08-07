#
# Settings class for PyBaMM
#


class Settings:
    _debug_mode = False
    _simplify = True
    _min_max_mode = "exact"
    _min_max_smoothing = 10
    _heaviside_smoothing = "exact"
    _abs_smoothing = "exact"
    max_words_in_line = 4
    max_y_value = 1e5
    step_start_offset = 1e-9
    tolerances = {
        "D_e__c_e": 10,  # dimensional
        "kappa_e__c_e": 10,  # dimensional
        "chi__c_e": 1e-2,  # dimensionless
        "D__c_s": 1e-10,  # dimensionless
        "U__c_s": 1e-10,  # dimensionless
        "j0__c_e": 1e-8,  # dimensionless
        "j0__c_s": 1e-8,  # dimensionless
        "macinnes__c_e": 1e-15,  # dimensionless
    }

    @property
    def debug_mode(self):
        return self._debug_mode

    @debug_mode.setter
    def debug_mode(self, value: bool):
        if not isinstance(value, bool):
            raise TypeError(f"{value} must be of type bool")
        self._debug_mode = value

    @property
    def simplify(self):
        return self._simplify

    @simplify.setter
    def simplify(self, value: bool):
        if not isinstance(value, bool):
            raise TypeError(f"{value} must be of type bool")
        self._simplify = value

    def set_smoothing_parameters(self, k):
        """Helper function to set all smoothing parameters"""
        if k == "exact":
            self.min_max_mode = "exact"
        else:
            self.min_max_smoothing = k
            self.min_max_mode = "soft"
        self.heaviside_smoothing = k
        self.abs_smoothing = k

    @staticmethod
    def check_k(k):
        if k != "exact" and k <= 0:
            raise ValueError(
                "Smoothing parameter must be 'exact' or a strictly positive number"
            )

    @property
    def min_max_mode(self):
        return self._min_max_mode

    @min_max_mode.setter
    def min_max_mode(self, mode):
        if mode not in ["exact", "soft", "smooth"]:
            raise ValueError("Smoothing mode must be 'exact', 'soft', or 'smooth'")
        self._min_max_mode = mode

    @property
    def min_max_smoothing(self):
        return self._min_max_smoothing

    @min_max_smoothing.setter
    def min_max_smoothing(self, k):
        if self._min_max_mode == "soft" and k <= 0:
            raise ValueError("Smoothing parameter must be a strictly positive number")
        if self._min_max_mode == "smooth" and k < 1:
            raise ValueError("Smoothing parameter must be greater than 1")
        self._min_max_smoothing = k

    @property
    def heaviside_smoothing(self):
        return self._heaviside_smoothing

    @heaviside_smoothing.setter
    def heaviside_smoothing(self, k):
        self.check_k(k)
        self._heaviside_smoothing = k

    @property
    def abs_smoothing(self):
        return self._abs_smoothing

    @abs_smoothing.setter
    def abs_smoothing(self, k):
        self.check_k(k)
        self._abs_smoothing = k


settings = Settings()
