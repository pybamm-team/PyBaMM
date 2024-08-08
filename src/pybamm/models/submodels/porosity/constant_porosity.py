#
# Class for constant porosity
#
import pybamm

from .base_porosity import BaseModel


class Constant(BaseModel):
    """Submodel for constant porosity

    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel
    """

    def get_fundamental_variables(self):
        eps_dict = {}
        depsdt_dict = {}
        for domain in self.options.whole_cell_domains:
            eps_dict[domain] = self.param.domain_params[domain.split()[0]].epsilon_init
            depsdt_dict[domain] = pybamm.FullBroadcast(0, domain, "current collector")

        variables = self._get_standard_porosity_variables(eps_dict)
        variables.update(self._get_standard_porosity_change_variables(depsdt_dict))

        return variables

    def set_events(self, variables):
        # No events since porosity is constant
        pass
