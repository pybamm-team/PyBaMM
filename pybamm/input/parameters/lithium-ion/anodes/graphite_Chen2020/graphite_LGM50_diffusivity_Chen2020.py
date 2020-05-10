from pybamm import exp, constants, Scalar


def graphite_LGM50_diffusivity_Chen2020(sto, T):
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
      sto: :class:`pybamm.Symbol`
         Electrode stochiometry
      T: :class:`pybamm.Symbol`
         Dimensional temperature

      Returns
      -------
      :class:`pybamm.Symbol`
         Solid diffusivity
   """

    D_ref = Scalar(3.3e-14, "[m2.s-1]")
    E_D_s = Scalar(42770, "[J.mol-1]")
    arrhenius = exp(E_D_s / constants.R * (1 / Scalar(298.15, "[K]") - 1 / T))

    return D_ref * arrhenius
