from pybamm import exp


def nmc_LGM50_diffusivity_Chen2020(sto, T, T_inf, E_D_s, R_g):
    """
   NMC diffusivity as a function of stoichiometry, in this case the
   diffusivity is taken to be a constant. The value is taken from [1].
   References
   ----------
   .. [1] Chang-Hui Chen, Ferran Brosa Planella, Kieran O’Regan, Dominika Gastol,
   W. Dhammika Widanage, and Emma Kendrick. "Development of Experimental Techniques
   for Parameterization of Multi-scale Lithium-ion Battery Models." Submitted for
   publication (2020).
   Parameters
   ----------
   sto : :class:`pybamm.Symbol`
      Electrode stochiometry
   T : :class:`pybamm.Symbol`
      Dimensional temperature [K]
   T_inf: :class:`pybamm.Symbol`
      Reference temperature [K]
   E_D_s: :class:`pybamm.Symbol`
      Solid diffusion activation energy [J.mol-1]
   R_g: :class:`pybamm.Symbol`
      The ideal gas constant [J.mol-1.K-1]
   Returns
   -------
   : :class:`pybamm.Symbol`
      Solid diffusivity [m2.s-1]
    """

    D_ref = 4e-15
    arrhenius = exp(E_D_s / R_g * (1 / T_inf - 1 / T))

    return D_ref * arrhenius
