def electrolyte_diffusivity_Nyman2008(c_e, T):
    """
    Diffusivity of LiPF6 in EC:EMC (3:7) as a function of ion concentration. The data
    comes from [1]

    References
    ----------
    .. [1] A. Nyman, M. Behm, and G. Lindbergh, "Electrochemical characterisation and
    modelling of the mass transport phenomena in LiPF6-EC-EMC electrolyte,"
    Electrochim. Acta, vol. 53, no. 22, pp. 6356–6365, 2008.

    Parameters
    ----------
    c_e: :class:`pybamm.Symbol`
        Dimensional electrolyte concentration
    T: :class:`pybamm.Symbol`
        Dimensional temperature

    Returns
    -------
    :class:`pybamm.Symbol`
        Solid diffusivity
    """

    D_c_e = 8.794e-11 * (c_e / 1000) ** 2 - 3.972e-10 * (c_e / 1000) + 4.862e-10

    # Nyman et al. (2008) does not provide temperature dependence

    return D_c_e


def electrolyte_conductivity_Nyman2008(c_e, T):
    """
    Conductivity of LiPF6 in EC:EMC (3:7) as a function of ion concentration. The data
    comes from [1].

    References
    ----------
    .. [1] A. Nyman, M. Behm, and G. Lindbergh, "Electrochemical characterisation and
    modelling of the mass transport phenomena in LiPF6-EC-EMC electrolyte,"
    Electrochim. Acta, vol. 53, no. 22, pp. 6356–6365, 2008.

    Parameters
    ----------
    c_e: :class:`pybamm.Symbol`
        Dimensional electrolyte concentration
    T: :class:`pybamm.Symbol`
        Dimensional temperature

    Returns
    -------
    :class:`pybamm.Symbol`
        Solid diffusivity
    """

    sigma_e = (
        0.1297 * (c_e / 1000) ** 3 - 2.51 * (c_e / 1000) ** 1.5 + 3.329 * (c_e / 1000)
    )

    # Nyman et al. (2008) does not provide temperature dependence

    return sigma_e


def get_parameter_values():
    """
    Example parameter values for use with MSMR models. The thermodynamic parameters
    are for Graphite and NMC622, and are taken from Table 1 of the paper

        Mark Verbrugge, Daniel Baker, Brian Koch, Xingcheng Xiao and Wentian Gu.
        Thermodynamic Model for Substitutional Materials: Application to Lithiated
        Graphite, Spinel Manganese Oxide, Iron Phosphate, and Layered
        Nickel-Manganese-Cobalt Oxide. Journal of The Electrochemical Society,
        164(11):3243-3253, 2017. doi:10.1149/2.0341708jes.

    The remaining value are based on a parameterization of the LG M50 cell, from the
    paper

        Chang-Hui Chen, Ferran Brosa Planella, Kieran O'Regan, Dominika Gastol, W.
        Dhammika Widanage, and Emma Kendrick. Development of Experimental Techniques for
        Parameterization of Multi-scale Lithium-ion Battery Models. Journal of The
        Electrochemical Society, 167(8):080534, 2020. doi:10.1149/1945-7111/ab9050.

    and references therein. Verbrugge et al. (2017) does not provide kinetic parameters
    so we set the reference exchange current density to 5 A.m-2 for the positive
    electrode reactions and 2.7 A.m-2 for the negative electrode reactions, which are
    the values used in the Chen et al. (2020) paper. We also assume that the
    exchange-current density is symmetric. Note: the 4th reaction in the positive
    electrode gave unphysical results so we set the reference exchange current density
    and symmetry factor to 1e6 and 1, respectively. The parameter values are intended
    to serve as an example set to use with the MSMR model and do not claim to match any
    experimental cycling data.
    """
    return {
        "chemistry": "lithium_ion",
        # cell
        "Negative electrode thickness [m]": 8.52e-05,
        "Separator thickness [m]": 1.2e-05,
        "Positive electrode thickness [m]": 7.56e-05,
        "Electrode height [m]": 0.065,
        "Electrode width [m]": 1.58,
        "Nominal cell capacity [A.h]": 5.0,
        "Current function [A]": 5.0,
        "Contact resistance [Ohm]": 0,
        # negative electrode
        "Number of reactions in negative electrode": 6,
        "U0_n_0": 0.08843,
        "X_n_0": 0.43336,
        "w_n_0": 0.08611,
        "a_n_0": 0.5,
        "j0_ref_n_0": 2.7,
        "U0_n_1": 0.12799,
        "X_n_1": 0.23963,
        "w_n_1": 0.08009,
        "a_n_1": 0.5,
        "j0_ref_n_1": 2.7,
        "U0_n_2": 0.14331,
        "X_n_2": 0.15018,
        "w_n_2": 0.72469,
        "a_n_2": 0.5,
        "j0_ref_n_2": 2.7,
        "U0_n_3": 0.16984,
        "X_n_3": 0.05462,
        "w_n_3": 2.53277,
        "a_n_3": 0.5,
        "j0_ref_n_3": 2.7,
        "U0_n_4": 0.21446,
        "X_n_4": 0.06744,
        "w_n_4": 0.09470,
        "a_n_4": 0.5,
        "j0_ref_n_4": 2.7,
        "U0_n_5": 0.36325,
        "X_n_5": 0.05476,
        "w_n_5": 5.97354,
        "a_n_5": 0.5,
        "j0_ref_n_5": 2.7,
        "Negative electrode conductivity [S.m-1]": 215.0,
        "Maximum concentration in negative electrode [mol.m-3]": 33133.0,
        "Negative particle diffusivity [m2.s-1]": 3.3e-14,
        "Negative electrode porosity": 0.25,
        "Negative electrode active material volume fraction": 0.75,
        "Negative particle radius [m]": 5.86e-06,
        "Negative electrode Bruggeman coefficient (electrolyte)": 1.5,
        "Negative electrode Bruggeman coefficient (electrode)": 0,
        "Negative electrode OCP entropic change [V.K-1]": 0.0,
        # positive electrode
        "Number of reactions in positive electrode": 4,
        "U0_p_0": 3.62274,
        "X_p_0": 0.13442,
        "w_p_0": 0.96710,
        "a_p_0": 0.5,
        "j0_ref_p_0": 5,
        "U0_p_1": 3.72645,
        "X_p_1": 0.32460,
        "w_p_1": 1.39712,
        "a_p_1": 0.5,
        "j0_ref_p_1": 5,
        "U0_p_2": 3.90575,
        "X_p_2": 0.21118,
        "w_p_2": 3.50500,
        "a_p_2": 0.5,
        "j0_ref_p_2": 5,
        "U0_p_3": 4.22955,
        "X_p_3": 0.32980,
        "w_p_3": 5.52757,
        "a_p_3": 1,
        "j0_ref_p_3": 1e6,
        "Positive electrode conductivity [S.m-1]": 0.18,
        "Maximum concentration in positive electrode [mol.m-3]": 63104.0,
        "Positive particle diffusivity [m2.s-1]": 4e-15,
        "Positive electrode porosity": 0.335,
        "Positive electrode active material volume fraction": 0.665,
        "Positive particle radius [m]": 5.22e-06,
        "Positive electrode Bruggeman coefficient (electrolyte)": 1.5,
        "Positive electrode Bruggeman coefficient (electrode)": 0,
        "Positive electrode OCP entropic change [V.K-1]": 0.0,
        # separator
        "Separator porosity": 0.47,
        "Separator Bruggeman coefficient (electrolyte)": 1.5,
        # electrolyte
        "Initial concentration in electrolyte [mol.m-3]": 1000.0,
        "Cation transference number": 0.2594,
        "Thermodynamic factor": 1.0,
        "Electrolyte diffusivity [m2.s-1]": electrolyte_diffusivity_Nyman2008,
        "Electrolyte conductivity [S.m-1]": electrolyte_conductivity_Nyman2008,
        # experiment
        "Reference temperature [K]": 298.15,
        "Total heat transfer coefficient [W.m-2.K-1]": 10.0,
        "Ambient temperature [K]": 298.15,
        "Number of electrodes connected in parallel to make a cell": 1.0,
        "Number of cells connected in series to make a battery": 1.0,
        "Lower voltage cut-off [V]": 2.8,
        "Upper voltage cut-off [V]": 4.2,
        "Open-circuit voltage at 0% SOC [V]": 2.8,
        "Open-circuit voltage at 100% SOC [V]": 4.2,
        "Initial temperature [K]": 298.15,
        "Initial voltage in negative electrode [V]": 0.01,
        "Initial voltage in positive electrode [V]": 4.19,
        # citations
        "citations": ["Verbrugge2017", "Baker2018", "Chen2020"],
    }
