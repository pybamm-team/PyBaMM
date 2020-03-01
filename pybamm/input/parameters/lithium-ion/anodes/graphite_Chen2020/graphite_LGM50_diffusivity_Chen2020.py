from pybamm import exp


def graphite_LGM50_diffusivity_Chen2020(sto, T, T_inf, E_D_s, R_g):
    """
      LG M50 Graphite diffusivity as a function of stochiometry, in this case the
      diffusivity is taken to be a constant. The value is taken from [1].
      References
      ----------
      .. [1] Chang-Hui Chen, Ferran Brosa Planella, Kieran O’Regan, Dominika Gastol, W.
      Dhammika Widanage, and Emma Kendrick. "Development of Experimental Techniques for
      Parameterization of Multi-scale Lithium-ion Battery Models." Submitted for
      publication (2020).
      Parameters
      ----------
      sto : :class:`pybamm.Symbol`
         Electrode stochiometry
      T : :class:`pybamm.Symbol`
         Dimensional temperature [K]
      T_inf: :class:`pybamm.Symbol`
         Reference temperature
      E_D_s: :class:`pybamm.Symbol`
         Solid diffusion activation energy
      R_g: :class:`pybamm.Symbol`
         The ideal gas constant [J.mol-1.K-1]
      Returns
      -------
      : :class:`pybamm.Symbol`
         Solid diffusivity
   """

    D_ref = 3.3e-14
    arrhenius = exp(E_D_s / R_g * (1 / T_inf - 1 / T))

    return D_ref * arrhenius
