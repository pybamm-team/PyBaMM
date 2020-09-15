import pybamm
from .solvent_diffusion_limited import SolventDiffusionLimited


class SEIonCracks(SolventDiffusionLimited):
    """
    Class for solvent-diffusion limited SEI growth in cracks, where the exposed area
    depends on the crack length.

    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel
    domain : str
        The domain of the model either 'Negative' or 'Positive'

    **Extends:** :class:`pybamm.sei.SolventDiffusionLimited`
    """

    def __init__(self, param, domain):
        super().__init__(param, domain)

        # Reaction name and icd are updated, but not the variable "self.reaction"
        # as this is still a SEI reaction, after all.
        self.reaction_name = " sei-cracks"
        self.reaction_icd = "Sei-cracks interfacial current density"

    def get_fundamental_variables(self):
        """The SEI growth in cracks has no fundamental variables.

        The usual L_inner and L_outer now depend on the crack length, so this method
        has to be overwritten and the creation of those variables, coupled to the
        crack length, done in the 'get_coupled_variables' method."""
        domain = self.domain.lower() + " electrode"
        L_inner_cr = pybamm.Variable(
            f"Inner {domain}{self.reaction_name} thickness",
            domain=self.domain.lower() + " electrode",
            auxiliary_domains={"secondary": "current collector"},
        )
        L_outer_cr = pybamm.Variable(
            f"Outer {domain}{self.reaction_name} thickness",
            domain=self.domain.lower() + " electrode",
            auxiliary_domains={"secondary": "current collector"},
        )
        variables = {
            f"Inner {domain}{self.reaction_name} thickness": L_inner_cr,
            f"Outer {domain}{self.reaction_name} thickness": L_outer_cr,
        }
        return variables

    def get_coupled_variables(self, variables):
        """Define the coupled variables of the submodel.

        TODO: Weilong, replace the information commented below for whatever makes sense
        TODO: to calculate the SEI thicknesses out of the crack length. You will need
        TODO: to ensure they are defined in the correct domains - probably the same
        TODO: than L_inner and L_outer.
        """
        # from here -->
        domain = self.domain.lower() + " electrode"
        L_inner_cr = variables[f"Inner {domain}{self.reaction_name} thickness"]
        L_outer_cr = variables[f"Outer {domain}{self.reaction_name} thickness"]

        variables.update(self._get_standard_thickness_variables(L_inner_cr, L_outer_cr))
        variables.update(self._get_standard_concentraion_variables(variables))
        # Now we get the normal coupled variables for the parent submodel
        return super().get_coupled_variables(variables)
