import pybamm
from .solvent_diffusion_limited import SolventDiffusionLimited


class SolvenDifussionLimitedInCracks(SolventDiffusionLimited):
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
        return {}

    def get_coupled_variables(self, variables):
        """Define the coupled variables of the submodel.

        TODO: Weilong, replace the information commented below for whatever makes sense
        TODO: to calculate the SEI thicknesses out of the crack length. You will need
        TODO: to ensure they are defined in the correct domains - probably the same
        TODO: than L_inner and L_outer.
        """
        # from here -->
        l_crack = variables["Name of the crack length variable"]
        L_inner_cracks = l_crack
        L_outer_cracks = l_crack
        # <-- to here

        variables.update(
            self._get_standard_thickness_variables(L_inner_cracks, L_outer_cracks)
        )
        variables.update(self._get_standard_concentraion_variables(variables))

        # Now we get the normal coupled variables for the parent submodel
        return super().get_coupled_variables(variables)
