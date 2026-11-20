#
# Base class for transport_efficiency
#
import pybamm


class BaseModel(pybamm.BaseSubModel):
    """Base class for transport_efficiency

    The transport efficiency of the electrolyte is a function of the porosity and
    that of the electrode a function of the solid volume fraction (one minus the
    porosity), which subclasses define in :meth:`_transport_efficiency`.

    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel
    component : str
        The material for the model ('Electrolyte' or 'Electrode').
    options : dict, optional
        A dictionary of options to be passed to the model.
    """

    def __init__(self, param, component, options=None):
        super().__init__(param, options=options)
        self.component = component

    def build(self):
        tor_dict = {}
        for domain in self.options.whole_cell_domains:
            if self.component == "Electrode" and domain == "separator":
                tor_dict[domain] = pybamm.FullBroadcast(
                    0, "separator", "current collector"
                )
                continue
            eps_k = self.coupled_variable(
                f"{domain.capitalize()} porosity",
                domain=domain,
                auxiliary_domains={"secondary": "current collector"},
            )
            volume_fraction = eps_k if self.component == "Electrolyte" else 1 - eps_k
            domain_param = self.param.domain_params[domain.split()[0]]
            tor_dict[domain] = self._transport_efficiency(volume_fraction, domain_param)

        self.variables.update(
            self._get_standard_transport_efficiency_variables(tor_dict)
        )

    def _transport_efficiency(self, volume_fraction, domain_param):
        """
        Transport efficiency of the component in one domain.

        Parameters
        ----------
        volume_fraction : :class:`pybamm.Symbol`
            Volume fraction of the component in the domain.
        domain_param : parameter class
            The parameters of the domain.

        Returns
        -------
        :class:`pybamm.Symbol`
        """
        raise NotImplementedError

    def _get_standard_transport_efficiency_variables(self, tor_dict):
        component = self.component.lower()

        tor = pybamm.concatenation(*tor_dict.values())

        variables = {f"{self.component} transport efficiency": tor}

        for domain, tor_k in tor_dict.items():
            domain = domain.split()[0]
            Domain = domain.capitalize()
            tor_k_av = pybamm.x_average(tor_k)

            variables.update(
                {
                    f"{Domain} {component} transport efficiency": tor_k,
                    f"X-averaged {domain} {component} transport efficiency": tor_k_av,
                }
            )

        # Override print_name
        tor.print_name = r"\mathcal{B}"

        return variables
