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
                    raise AttributeError(
                        f"param.{name} does not exist. It may have been renamed to "
                        f"param.{domain}.{name_without_domain}"
                    )
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
