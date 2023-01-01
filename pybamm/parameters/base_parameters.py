#
# Base parameters class
#
import pybamm
import warnings


class BaseParameters:
    """
    Overload the `__setattr__` method to record what the variable was called.
    """

    def __getattribute__(self, name):
        """
        Raise more informative error to users when they try to access a
        non-existent attribute, which may have recently changed name
        """
        try:
            return super().__getattribute__(name)
        except AttributeError as e:
            if name == "cap_init":
                warnings.warn("Parameter 'cap_init' has been renamed to 'Q_init'")
                return self.Q_init
            for domain in ["n", "s", "p"]:
                if f"_{domain}_" in name or name.endswith(f"_{domain}"):
                    name_without_domain = name.replace(f"_{domain}_", "_").replace(
                        f"_{domain}", ""
                    )
                    if hasattr(self, domain):
                        self_domain = getattr(self, domain)
                        if hasattr(self_domain, name_without_domain):
                            raise AttributeError(
                                f"param.{name} does not exist. It has been renamed to "
                                f"param.{domain}.{name_without_domain}"
                            )
                        elif hasattr(self_domain, "prim") and hasattr(
                            self_domain.prim, name_without_domain
                        ):
                            raise AttributeError(
                                f"param.{name} does not exist. It has been renamed to "
                                f"param.{domain}.prim.{name_without_domain}"
                            )
                        else:
                            raise e
                    else:
                        raise e
            raise e

    def __setattr__(self, name, value):
        if hasattr(self, "domain"):
            d = self.domain[0]
            print_name = f"{name}_{d}"
        else:
            print_name = name
        if isinstance(value, pybamm.Symbol):
            value.print_name = print_name
        super().__setattr__(name, value)

    @property
    def options(self):
        return self._options

    @options.setter
    def options(self, extra_options):
        if extra_options is None or type(extra_options) == dict:
            self._options = pybamm.BatteryModelOptions(extra_options)
        else:
            self._options = extra_options

    @property
    def domain(self):
        return self._domain

    @domain.setter
    def domain(self, domain):
        self._domain = domain
        if domain is not None:
            self._Domain = domain.capitalize()

    @property
    def domain_Domain(self):
        return self.domain, self._Domain

    def set_phase_name(self):
        if (
            self.phase == "primary"
            and getattr(self.main_param.options, self.domain)["particle phases"] == "1"
        ):
            # Only one phase, no need to distinguish between
            # "primary" and "secondary"
            self.phase_name = ""
            self.phase_prefactor = ""
        else:
            # add a space so that we can use "" or (e.g.) "primary " interchangeably
            # when naming variables
            self.phase_name = self.phase + " "
            self.phase_prefactor = self.phase.capitalize() + ": "


class NullParameters:
    def __getattribute__(self, name):
        "Returns 0 for some parameters that aren't found by __getattribute__"
        if name in [
            "epsilon_s",
            "Q_init",
            "n_Li_init",
            "Q_Li_init",
            "R_typ",
            "j_scale",
        ]:
            return pybamm.Scalar(0)
        else:
            return super().__getattribute__(name)

    def _set_dimensional_parameters(self):
        pass

    def _set_scales(self):
        pass

    def _set_dimensionless_parameters(self):
        pass
