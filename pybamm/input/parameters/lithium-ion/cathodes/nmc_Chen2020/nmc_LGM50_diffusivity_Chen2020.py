from pybamm import exp, constants


def nmc_LGM50_diffusivity_Chen2020(sto, T):
    """
       NMC diffusivity as a function of stoichiometry, in this case the
       diffusivity is taken to be a constant. The value is taken from [1].

       References
       ----------
      .. [1] Chang-Hui Chen, Ferran Brosa Planella, Kieran O’Regan, Dominika Gastol, W.
      Dhammika Widanage, and Emma Kendrick. "Development of Experimental Techniques for
      Parameterization of Multi-scale Lithium-ion Battery Models." Journal of the
      Electrochemical Society 167 (2020): 080534.

       Parameters
       ----------
       sto: :class:`pybamm.Symbol`
         Electrode stochiometry
       T: :class:`pybamm.Symbol`
          Dimensional temperature

       Returns
       -------
       :class:`pybamm.Symbol`
          Solid diffusivity
    """

    D_ref = 4e-15
    E_D_s = 0  # to be implemented
    arrhenius = exp(E_D_s / constants.R * (1 / 298.15 - 1 / T))

    return D_ref * arrhenius
