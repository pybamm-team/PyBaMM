#
# Base class for electrolyte diffusion employing stefan-maxwell
#
import pybamm


class FullStefanMaxwellDiffusion(pybamm.BaseStefanMaxwellDiffusion):
    """Class for conservation of mass in the electrolyte employing the
    Stefan-Maxwell constitutive equations. (Full refers to unreduced by
    asymptotic methods)

    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel

    *Extends:* :class:`pybamm.BaseStefanMaxwellDiffusion`
    """

    def __init__(self, param, domain):
        super().__init__(param)
        self._domain = domain

    def get_fundamental_variables(self):
        """
        Returns the variables in the submodel which can be stated independent of 
        variables stated in other submodels
        """

        c_e = pybamm.standard_variables.c_e

        variables = self._get_standard_concentration_variables(c_e)

        return variables

    def get_coupled_variables(self, variables):

        epsilon = variables["Porosity"]
        c_e = variables["Electrolyte concentration"]
        i_e = variables["Electrolyte current density"]
        v_box = variables["Volume-averaged velocity"]

