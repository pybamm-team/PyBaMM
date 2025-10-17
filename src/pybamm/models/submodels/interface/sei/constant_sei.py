#
# Class for constant SEI thickness
#
import pybamm

from .base_sei import BaseModel


class ConstantSEI(BaseModel):
    """
    Class for SEI with constant concentration.

    Note that there is no SEI current, so we don't need to update the "sum of
    interfacial current densities" variables from
    :class:`pybamm.interface.BaseInterface`

    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel
    options : dict
        A dictionary of options to be passed to the model.
    phase : str, optional
        Phase of the particle (default is "primary")
    """

    def __init__(self, param, domain, options, phase="primary"):
        super().__init__(param, domain, options=options, phase=phase)
        if self.options.electrode_types[domain] == "planar":
            self.reaction_loc = "interface"
        else:
            self.reaction_loc = "full electrode"

    def get_fundamental_variables(self):
        domain = self.domain.lower()
        L_sei_0 = self.phase_param.L_sei_0
        V_bar_sei = self.phase_param.V_bar_sei
        # Constant concentration
        if self.reaction_loc == "interface":
            # c_sei is an interfacial quantity [mol.m-2]
            c_sei = L_sei_0 / V_bar_sei
        else:
            # c_sei is a bulk quantity [mol.m-3]
            c_sei = L_sei_0 * self.phase_param.a_typ / V_bar_sei
        variables = self._get_standard_concentration_variables(c_sei)

        # Reactions
        if self.reaction_loc == "interface":
            zero = pybamm.PrimaryBroadcast(pybamm.Scalar(0), "current collector")
        else:
            zero = pybamm.FullBroadcast(
                pybamm.Scalar(0), f"{domain} electrode", "current collector"
            )
        variables.update(self._get_standard_reaction_variables(zero))

        return variables
