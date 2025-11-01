import pybamm

from .base_submodel import BaseSubModel


class BaseBatterySubModel(BaseSubModel):
    """
    Base class for all battery-specific submodels.
    Contains domain checks and other logic specific to battery modeling.
    """

    ok_domain_list = [
        "negative",
        "separator",
        "positive",
        "negative electrode",
        "negative electrolyte",
        "separator electrolyte",
        "positive electrode",
        "positive electrolyte",
        None,
    ]

    def __init__(
        self,
        param,
        domain=None,
        name="Unnamed battery submodel",
        external=False,
        options=None,
        phase=None,
    ):
        # normalize domain
        domain_lower = domain.lower() if isinstance(domain, str) else domain

        if domain_lower not in self.ok_domain_list:
            raise pybamm.DomainError(
                f"Invalid domain '{domain}' for battery submodel. "
                f"Allowed: {self.ok_domain_list}"
            )

        super().__init__(param, domain, name, external, options, phase)
