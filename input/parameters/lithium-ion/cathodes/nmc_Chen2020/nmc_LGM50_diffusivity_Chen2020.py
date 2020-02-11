from pybamm import exp


def nmc_LGM50_diffusivity_Chen2020(sto, T, T_inf, E_D_s, R_g):
    """
       NMC diffusivity as a function of stoichiometry, in this case the
       diffusivity is taken to be a constant. The value is taken from [1].
       References
       ----------
       .. [1] Work in progress
       Parameters
       ----------
       sto: :class: `numpy.Array`
         Electrode stochiometry
       T: :class: `numpy.Array`
          Dimensional temperature
       T_inf: double
          Reference temperature
       E_D_s: double
          Solid diffusion activation energy
       R_g: double
          The ideal gas constant
       Returns
       -------
       : double
          Solid diffusivity
    """

    D_ref = 4e-15
    arrhenius = exp(E_D_s / R_g * (1 / T_inf - 1 / T))

    return D_ref * arrhenius
