import autograd.numpy as np


def graphite_mcmb2528_diffusivity_Dualfoil(sto, T, T_inf, E_D_s, R_g):
    """
      Graphite MCMB 2528 diffusivity as a function of stochiometry, in this case the
      diffusivity is taken to be a constant. The value is taken from Dualfoil [1].

      References
      ----------
      .. [1] http://www.cchem.berkeley.edu/jsngrp/fortran.html

      Parameters
      ----------
      sto: :class: `pybamm.Symbol`
         Electrode stochiometry
      T: :class: `pybamm.Symbol`
         Dimensional temperature
      T_inf: :class: `pybamm.Parameter`
         Reference temperature
      E_D_s: :class: `pybamm.Parameter`
         Solid diffusion activation energy
      R_g: :class: `pybamm.Parameter`
         The ideal gas constant

      Returns
      -------
      :`pybamm.Symbol`
         Solid diffusivity
   """

    D_ref = 3.9 * 10 ** (-14)
    arrhenius = np.exp(E_D_s / R_g * (1 / T_inf - 1 / T))

    correct_shape = 0 * sto

    return D_ref * arrhenius + correct_shape

