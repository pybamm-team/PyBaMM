from pybamm import exp


def graphite_diffusivity_Kim2011(sto, T, T_inf, E_D_s, R_g):
    """
        Graphite diffusivity [1].

        References
        ----------
        .. [1] Kim, G. H., Smith, K., Lee, K. J., Santhanagopalan, S., & Pesaran, A.
        (2011). Multi-domain modeling of lithium-ion batteries encompassing
        multi-physics in varied length scales. Journal of The Electrochemical
        Society, 158(8), A955-A969.

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

    D_ref = 9 * 10 ** (-14)
    arrhenius = exp(E_D_s / R_g * (1 / T_inf - 1 / T))

    return D_ref * arrhenius
