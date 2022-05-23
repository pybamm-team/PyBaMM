#
# Base parameters class
#
import pybamm


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
            d = self.domain.lower()[0]
            print_name = f"{name}_{d}"
        else:
            print_name = name
        if isinstance(value, pybamm.Symbol):
            value.print_name = print_name
        super().__setattr__(name, value)


class NullParameters:
    def __getattr__(self, name):
        "Returns 0 for every parameter that wasn't found by __getattribute__"
        return pybamm.Scalar(0)

    def _set_dimensional_parameters(self):
        pass

    def _set_scales(self):
        pass

    def _set_dimensionless_parameters(self):
        pass
