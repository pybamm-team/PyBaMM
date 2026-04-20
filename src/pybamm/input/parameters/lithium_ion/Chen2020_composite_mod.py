import os

import numpy as np

import pybamm

def volume_change_Ai2020(sto):
    """
    Particle volume change as a function of stoichiometry [1, 2].

    References
    ----------
     .. [1] > Ai, W., Kraft, L., Sturm, J., Jossen, A., & Wu, B. (2020).
     Electrochemical Thermal-Mechanical Modelling of Stress Inhomogeneity in
     Lithium-Ion Pouch Cells. Journal of The Electrochemical Society, 167(1), 013512
      DOI: 10.1149/2.0122001JES.
     .. [2] > Rieger, B., Erhard, S. V., Rumpf, K., & Jossen, A. (2016).
     A new method to model the thickness change of a commercial pouch cell
     during discharge. Journal of The Electrochemical Society, 163(8), A1566-A1575.

    Parameters
    ----------
    sto: :class:`pybamm.Symbol`
        Electrode stoichiometry, dimensionless
        should be R-averaged particle concentration
    Returns
    -------
    t_change:class:`pybamm.Symbol`
        volume change, dimensionless, normalised by particle volume
    """
    omega = pybamm.Parameter("Positive electrode partial molar volume [m3.mol-1]")
    c_s_max = pybamm.Parameter("Maximum concentration in positive electrode [mol.m-3]")
    t_change = omega * c_s_max * sto
    return t_change

def graphite_volume_change_Ai2020(sto):
    """
    Graphite particle volume change as a function of stoichiometry [1, 2].

    References
    ----------
     .. [1] Ai, W., Kraft, L., Sturm, J., Jossen, A., & Wu, B. (2020).
     Electrochemical Thermal-Mechanical Modelling of Stress Inhomogeneity in
     Lithium-Ion Pouch Cells. Journal of The Electrochemical Society, 167(1), 013512
      DOI: 10.1149/2.0122001JES.
     .. [2] Rieger, B., Erhard, S. V., Rumpf, K., & Jossen, A. (2016).
     A new method to model the thickness change of a commercial pouch cell
     during discharge. Journal of The Electrochemical Society, 163(8), A1566-A1575.

    Parameters
    ----------
    sto: :class:`pybamm.Symbol`
        Electrode stoichiometry, dimensionless
        should be R-averaged particle concentration
    Returns
    -------
    t_change:class:`pybamm.Symbol`
        volume change, dimensionless, normalised by particle volume
    """
    p1 = 145.907
    p2 = -681.229
    p3 = 1334.442
    p4 = -1415.710
    p5 = 873.906
    p6 = -312.528
    p7 = 60.641
    p8 = -5.706
    p9 = 0.386
    p10 = -4.966e-05
    t_change = (
        p1 * sto**9
        + p2 * sto**8
        + p3 * sto**7
        + p4 * sto**6
        + p5 * sto**5
        + p6 * sto**4
        + p7 * sto**3
        + p8 * sto**2
        + p9 * sto
        + p10
    )
    return t_change


def graphite_cracking_rate_Ai2020(T_dim):
    """
    Graphite particle cracking rate as a function of temperature [1, 2].

    References
    ----------
     .. [1] Ai, W., Kraft, L., Sturm, J., Jossen, A., & Wu, B. (2020).
     Electrochemical Thermal-Mechanical Modelling of Stress Inhomogeneity in
     Lithium-Ion Pouch Cells. Journal of The Electrochemical Society, 167(1), 013512
      DOI: 10.1149/2.0122001JES.
     .. [2] Deshpande, R., Verbrugge, M., Cheng, Y. T., Wang, J., & Liu, P. (2012).
     Battery cycle life prediction with coupled chemical degradation and fatigue
     mechanics. Journal of the Electrochemical Society, 159(10), A1730.

    Parameters
    ----------
    T_dim: :class:`pybamm.Symbol`
        temperature, [K]

    Returns
    -------
    k_cr: :class:`pybamm.Symbol`
        cracking rate, [m/(Pa.m0.5)^m_cr]
        where m_cr is another Paris' law constant
    """
    k_cr = pybamm.Parameter("Negative electrode cracking rate constant [m/(Pa.m0.5)^m_cr]")
    Eac_cr = 0  # to be implemented
    arrhenius = np.exp(Eac_cr / pybamm.constants.R * (1 / T_dim - 1 / 298.15))
    return k_cr * arrhenius

def cracking_rate_Ai2020(T_dim):
    """
    Particle cracking rate as a function of temperature [1, 2].

    References
    ----------
     .. [1] > Ai, W., Kraft, L., Sturm, J., Jossen, A., & Wu, B. (2020).
     Electrochemical Thermal-Mechanical Modelling of Stress Inhomogeneity in
     Lithium-Ion Pouch Cells. Journal of The Electrochemical Society, 167(1), 013512
      DOI: 10.1149/2.0122001JES.
     .. [2] > Deshpande, R., Verbrugge, M., Cheng, Y. T., Wang, J., & Liu, P. (2012).
     Battery cycle life prediction with coupled chemical degradation and fatigue
     mechanics. Journal of the Electrochemical Society, 159(10), A1730.

    Parameters
    ----------
    T: :class:`pybamm.Symbol`
        temperature, [K]

    Returns
    -------
    k_cr: :class:`pybamm.Symbol`
        cracking rate, [m/(Pa.m0.5)^m_cr]
        where m_cr is another Paris' law constant
    """
    k_cr = 3.9e-20
    k_cr = pybamm.Parameter("Positive electrode cracking rate constant [m/(Pa.m0.5)^m_cr]")
    Eac_cr = 0  # to be implemented
    arrhenius = np.exp(Eac_cr / pybamm.constants.R * (1 / T_dim - 1 / 298.15))
    return k_cr * arrhenius

 #changed until here

def graphite_LGM50_diffusivity_Chen2020(sto, T):
    """
    LG M50 Graphite (primary) diffusivity as a function of stoichiometry and
    temperature, using an Arrhenius correction. The reference value and activation
    energy are read from the parameter set.

    Parameters
    ----------
    sto: :class:`pybamm.Symbol`
       Electrode stoichiometry
    T: :class:`pybamm.Symbol`
       Dimensional temperature

    Returns
    -------
    :class:`pybamm.Symbol`
       Solid diffusivity
    """
    D_ref = pybamm.Parameter("Primary: Negative particle diffusivity constant [m2.s-1]")
    E_D_s = pybamm.Parameter(
        "Primary: Negative particle diffusivity activation energy [J.mol-1]"
    )
    arrhenius = np.exp(E_D_s / pybamm.constants.R * (1 / 298.15 - 1 / T))
    return D_ref * arrhenius


def silicon_LGM50_diffusivity_Chen2020(sto, T):
    """
    Silicon (secondary) diffusivity as a function of stoichiometry and temperature,
    using an Arrhenius correction. The reference value and activation energy are read
    from the parameter set.

    Parameters
    ----------
    sto: :class:`pybamm.Symbol`
       Electrode stoichiometry
    T: :class:`pybamm.Symbol`
       Dimensional temperature

    Returns
    -------
    :class:`pybamm.Symbol`
       Solid diffusivity
    """
    D_ref = pybamm.Parameter(
        "Secondary: Negative particle diffusivity constant [m2.s-1]"
    )
    E_D_s = pybamm.Parameter(
        "Secondary: Negative particle diffusivity activation energy [J.mol-1]"
    )
    arrhenius = np.exp(E_D_s / pybamm.constants.R * (1 / 298.15 - 1 / T))
    return D_ref * arrhenius


def nmc_LGM50_diffusivity_Chen2020(sto, T):
    """
    NMC diffusivity as a function of stoichiometry and temperature, using an Arrhenius
    correction. The reference value and activation energy are read from the parameter
    set.

    Parameters
    ----------
    sto: :class:`pybamm.Symbol`
       Electrode stoichiometry
    T: :class:`pybamm.Symbol`
       Dimensional temperature

    Returns
    -------
    :class:`pybamm.Symbol`
       Solid diffusivity
    """
    D_ref = pybamm.Parameter("Positive particle diffusivity constant [m2.s-1]")
    E_D_s = pybamm.Parameter("Positive particle diffusivity activation energy [J.mol-1]")
    arrhenius = np.exp(E_D_s / pybamm.constants.R * (1 / 298.15 - 1 / T))
    return D_ref * arrhenius


def graphite_LGM50_electrolyte_exchange_current_density_Chen2020(
    c_e, c_s_surf, c_s_max, T
):
    """
    Exchange-current density for Butler-Volmer reactions between graphite and LiPF6 in
    EC:DMC.

    References
    ----------
    .. [1] Chang-Hui Chen, Ferran Brosa Planella, Kieran O’Regan, Dominika Gastol, W.
    Dhammika Widanage, and Emma Kendrick. "Development of Experimental Techniques for
    Parameterization of Multi-scale Lithium-ion Battery Models." Journal of the
    Electrochemical Society 167 (2020): 080534.

    Parameters
    ----------
    c_e : :class:`pybamm.Symbol`
        Electrolyte concentration [mol.m-3]
    c_s_surf : :class:`pybamm.Symbol`
        Particle concentration [mol.m-3]
    c_s_max : :class:`pybamm.Symbol`
        Maximum particle concentration [mol.m-3]
    T : :class:`pybamm.Symbol`
        Temperature [K]

    Returns
    -------
    :class:`pybamm.Symbol`
        Exchange-current density [A.m-2]
    """
    m_ref = pybamm.Parameter("Primary: Negative electrode kinetic rate constant [A.m-2]")
    E_r = pybamm.Parameter(
        "Primary: Negative electrode exchange-current density activation energy [J.mol-1]"
    )
    arrhenius = np.exp(E_r / pybamm.constants.R * (1 / 298.15 - 1 / T))

    return m_ref * arrhenius * c_e**0.5 * c_s_surf**0.5 * (c_s_max - c_s_surf) ** 0.5


def silicon_ocp_lithiation_Mark2016(sto):
    """
    silicon Open-circuit Potential (OCP) as a a function of the
    stoichiometry. The fit is taken from the Enertech cell [1], which is only accurate
    for 0 < sto < 1.

    References
    ----------
    .. [1] Verbrugge M, Baker D, Xiao X. Formulation for the treatment of multiple
    electrochemical reactions and associated speciation for the Lithium-Silicon
    electrode[J]. Journal of The Electrochemical Society, 2015, 163(2): A262.

    Parameters
    ----------
    sto: double
       stoichiometry of material (li-fraction)

    Returns
    -------
    :class:`pybamm.Symbol`
        OCP [V]
    """
    p1 = -96.63
    p2 = 372.6
    p3 = -587.6
    p4 = 489.9
    p5 = -232.8
    p6 = 62.99
    p7 = -9.286
    p8 = 0.8633

    U_lithiation = (
        p1 * sto**7
        + p2 * sto**6
        + p3 * sto**5
        + p4 * sto**4
        + p5 * sto**3
        + p6 * sto**2
        + p7 * sto
        + p8
    )
    return U_lithiation


def silicon_ocp_delithiation_Mark2016(sto):
    """
    silicon Open-circuit Potential (OCP) as a a function of the
    stoichiometry. The fit is taken from the Enertech cell [1], which is only accurate
    for 0 < sto < 1.

    References
    ----------
    .. [1] Verbrugge M, Baker D, Xiao X. Formulation for the treatment of multiple
    electrochemical reactions and associated speciation for the Lithium-Silicon
    electrode[J]. Journal of The Electrochemical Society, 2015, 163(2): A262.

    Parameters
    ----------
    sto: double
       stoichiometry of material (li-fraction)

    Returns
    -------
    :class:`pybamm.Symbol`
        OCP [V]
    """
    p1 = -51.02
    p2 = 161.3
    p3 = -205.7
    p4 = 140.2
    p5 = -58.76
    p6 = 16.87
    p7 = -3.792
    p8 = 0.9937

    U_delithiation = (
        p1 * sto**7
        + p2 * sto**6
        + p3 * sto**5
        + p4 * sto**4
        + p5 * sto**3
        + p6 * sto**2
        + p7 * sto
        + p8
    )
    return U_delithiation


def silicon_ocp_average_Mark2016(sto):
    return (
        silicon_ocp_lithiation_Mark2016(sto) + silicon_ocp_delithiation_Mark2016(sto)
    ) / 2


def silicon_LGM50_electrolyte_exchange_current_density_Chen2020(
    c_e, c_s_surf, c_s_max, T
):
    """
    Exchange-current density for Butler-Volmer reactions between silicon and LiPF6 in
    EC:DMC.

    References
    ----------
    .. [1] Chang-Hui Chen, Ferran Brosa Planella, Kieran O’Regan, Dominika Gastol, W.
    Dhammika Widanage, and Emma Kendrick. "Development of Experimental Techniques for
    Parameterization of Multi-scale Lithium-ion Battery Models." Journal of the
    Electrochemical Society 167 (2020): 080534.

    Parameters
    ----------
    c_e : :class:`pybamm.Symbol`
        Electrolyte concentration [mol.m-3]
    c_s_surf : :class:`pybamm.Symbol`
        Particle concentration [mol.m-3]
    c_s_max : :class:`pybamm.Symbol`
        Maximum particle concentration [mol.m-3]
    T : :class:`pybamm.Symbol`
        Temperature [K]

    Returns
    -------
    :class:`pybamm.Symbol`
        Exchange-current density [A.m-2]
    """

    m_ref = pybamm.Parameter(
        "Secondary: Negative electrode kinetic rate constant [A.m-2]"
    )
    E_r = pybamm.Parameter(
        "Secondary: Negative electrode exchange-current density activation energy [J.mol-1]"
    )
    arrhenius = np.exp(E_r / pybamm.constants.R * (1 / 298.15 - 1 / T))

    return m_ref * arrhenius * c_e**0.5 * c_s_surf**0.5 * (c_s_max - c_s_surf) ** 0.5


def nmc_LGM50_ocp_Chen2020(sto):
    """
    LG M50 NMC open-circuit potential as a function of stoichiometry, fit taken
    from [1].

    References
    ----------
    .. [1] Chang-Hui Chen, Ferran Brosa Planella, Kieran O’Regan, Dominika Gastol, W.
    Dhammika Widanage, and Emma Kendrick. "Development of Experimental Techniques for
    Parameterization of Multi-scale Lithium-ion Battery Models." Journal of the
    Electrochemical Society 167 (2020): 080534.

    Parameters
    ----------
    sto: :class:`pybamm.Symbol`
        Electrode stoichiometry

    Returns
    -------
    :class:`pybamm.Symbol`
        Open-circuit potential
    """

    u_eq = (
        -0.8090 * sto
        + 4.4875
        - 0.0428 * np.tanh(18.5138 * (sto - 0.5542))
        - 17.7326 * np.tanh(15.7890 * (sto - 0.3117))
        + 17.5842 * np.tanh(15.9308 * (sto - 0.3120))
    )

    return u_eq


def nmc_LGM50_electrolyte_exchange_current_density_Chen2020(c_e, c_s_surf, c_s_max, T):
    """
    Exchange-current density for Butler-Volmer reactions between NMC and LiPF6 in
    EC:DMC.

    References
    ----------
    .. [1] Chang-Hui Chen, Ferran Brosa Planella, Kieran O’Regan, Dominika Gastol, W.
    Dhammika Widanage, and Emma Kendrick. "Development of Experimental Techniques for
    Parameterization of Multi-scale Lithium-ion Battery Models." Journal of the
    Electrochemical Society 167 (2020): 080534.

    Parameters
    ----------
    c_e : :class:`pybamm.Symbol`
        Electrolyte concentration [mol.m-3]
    c_s_surf : :class:`pybamm.Symbol`
        Particle concentration [mol.m-3]
    c_s_max : :class:`pybamm.Symbol`
        Maximum particle concentration [mol.m-3]
    T : :class:`pybamm.Symbol`
        Temperature [K]

    Returns
    -------
    :class:`pybamm.Symbol`
        Exchange-current density [A.m-2]
    """
    m_ref = pybamm.Parameter("Positive electrode kinetic rate constant [A.m-2]")
    E_r = pybamm.Parameter(
        "Positive electrode exchange-current density activation energy [J.mol-1]"
    )
    arrhenius = np.exp(E_r / pybamm.constants.R * (1 / 298.15 - 1 / T))

    return m_ref * arrhenius * c_e**0.5 * c_s_surf**0.5 * (c_s_max - c_s_surf) ** 0.5


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
#plating
def graphite_plating_exchange_current_density_OKane2020(c_e, c_Li, T):
    """
    Exchange-current density for Li plating reaction [A.m-2].
    References
    ----------
    .. [1] O’Kane, Simon EJ, Ian D. Campbell, Mohamed WJ Marzook, Gregory J. Offer, and
    Monica Marinescu. "Physical origin of the differential voltage minimum associated
    with lithium plating in Li-ion batteries." Journal of The Electrochemical Society
    167, no. 9 (2020): 090540.
    Parameters
    ----------
    c_e : :class:`pybamm.Symbol`
        Electrolyte concentration [mol.m-3]
    c_Li : :class:`pybamm.Symbol`
        Plated lithium concentration [mol.m-3]
    T : :class:`pybamm.Symbol`
        Temperature [K]
    Returns
    -------
    :class:`pybamm.Symbol`
        Exchange-current density [A.m-2]
    """

    k_plating = pybamm.Parameter("Primary: Lithium plating kinetic rate constant [m.s-1]")

    return pybamm.constants.F * k_plating * c_e


def graphite_stripping_exchange_current_density_OKane2020(c_e, c_Li, T):
    """
    Exchange-current density for Li stripping reaction [A.m-2].

    References
    ----------

    .. [1] O’Kane, Simon EJ, Ian D. Campbell, Mohamed WJ Marzook, Gregory J. Offer, and
    Monica Marinescu. "Physical origin of the differential voltage minimum associated
    with lithium plating in Li-ion batteries." Journal of The Electrochemical Society
    167, no. 9 (2020): 090540.

    Parameters
    ----------

    c_e : :class:`pybamm.Symbol`
        Electrolyte concentration [mol.m-3]
    c_Li : :class:`pybamm.Symbol`
        Plated lithium concentration [mol.m-3]
    T : :class:`pybamm.Symbol`
        Temperature [K]

    Returns
    -------

    :class:`pybamm.Symbol`
        Exchange-current density [A.m-2]
    """

    k_plating = pybamm.Parameter("Primary: Lithium plating kinetic rate constant [m.s-1]")

    return pybamm.constants.F * k_plating * c_Li


def graphite_SEI_limited_dead_lithium_OKane2022(L_sei):
    """
    Decay rate for dead lithium formation [s-1].
    References
    ----------
    .. [1] Simon E. J. O'Kane, Weilong Ai, Ganesh Madabattula, Diega Alonso-Alvarez,
    Robert Timms, Valentin Sulzer, Jaqueline Sophie Edge, Billy Wu, Gregory J. Offer
    and Monica Marinescu. "Lithium-ion battery degradation: how to model it."
    Physical Chemistry: Chemical Physics 24, no. 13 (2022): 7909-7922.
    Parameters
    ----------
    L_sei : :class:`pybamm.Symbol`
        Total SEI thickness [m]
    Returns
    -------
    :class:`pybamm.Symbol`
        Dead lithium decay rate [s-1]
    """

    gamma_0 = pybamm.Parameter("Primary: Dead lithium decay constant [s-1]")
    L_sei_0 = pybamm.Parameter("Primary: Initial SEI thickness [m]")

    gamma = gamma_0 * L_sei_0 / L_sei

    return gamma

def silicon_plating_exchange_current_density_OKane2020(c_e, c_Li, T):
    """
    Exchange-current density for Li plating reaction [A.m-2].
    References
    ----------
    .. [1] O’Kane, Simon EJ, Ian D. Campbell, Mohamed WJ Marzook, Gregory J. Offer, and
    Monica Marinescu. "Physical origin of the differential voltage minimum associated
    with lithium plating in Li-ion batteries." Journal of The Electrochemical Society
    167, no. 9 (2020): 090540.
    Parameters
    ----------
    c_e : :class:`pybamm.Symbol`
        Electrolyte concentration [mol.m-3]
    c_Li : :class:`pybamm.Symbol`
        Plated lithium concentration [mol.m-3]
    T : :class:`pybamm.Symbol`
        Temperature [K]
    Returns
    -------
    :class:`pybamm.Symbol`
        Exchange-current density [A.m-2]
    """

    k_plating = pybamm.Parameter("Secondary: Lithium plating kinetic rate constant [m.s-1]")

    return pybamm.constants.F * k_plating * c_e


def silicon_stripping_exchange_current_density_OKane2020(c_e, c_Li, T):
    """
    Exchange-current density for Li stripping reaction [A.m-2].

    References
    ----------

    .. [1] O’Kane, Simon EJ, Ian D. Campbell, Mohamed WJ Marzook, Gregory J. Offer, and
    Monica Marinescu. "Physical origin of the differential voltage minimum associated
    with lithium plating in Li-ion batteries." Journal of The Electrochemical Society
    167, no. 9 (2020): 090540.

    Parameters
    ----------

    c_e : :class:`pybamm.Symbol`
        Electrolyte concentration [mol.m-3]
    c_Li : :class:`pybamm.Symbol`
        Plated lithium concentration [mol.m-3]
    T : :class:`pybamm.Symbol`
        Temperature [K]

    Returns
    -------

    :class:`pybamm.Symbol`
        Exchange-current density [A.m-2]
    """

    k_plating = pybamm.Parameter("Secondary: Lithium plating kinetic rate constant [m.s-1]")

    return pybamm.constants.F * k_plating * c_Li


def silicon_SEI_limited_dead_lithium_OKane2022(L_sei):
    """
    Decay rate for dead lithium formation [s-1].
    References
    ----------
    .. [1] Simon E. J. O'Kane, Weilong Ai, Ganesh Madabattula, Diega Alonso-Alvarez,
    Robert Timms, Valentin Sulzer, Jaqueline Sophie Edge, Billy Wu, Gregory J. Offer
    and Monica Marinescu. "Lithium-ion battery degradation: how to model it."
    Physical Chemistry: Chemical Physics 24, no. 13 (2022): 7909-7922.
    Parameters
    ----------
    L_sei : :class:`pybamm.Symbol`
        Total SEI thickness [m]
    Returns
    -------
    :class:`pybamm.Symbol`
        Dead lithium decay rate [s-1]
    """

    gamma_0 = pybamm.Parameter("Secondary: Dead lithium decay constant [s-1]")
    L_sei_0 = pybamm.Parameter("Secondary:Initial SEI thickness [m]")

    gamma = gamma_0 * L_sei_0 / L_sei

    return gamma



# Load data in the appropriate format
path, _ = os.path.split(os.path.abspath(__file__))
graphite_ocp_Enertech_Ai2020_data = pybamm.parameters.process_1D_data(
    "graphite_ocp_Enertech_Ai2020.csv", path=path
)


def graphite_ocp_Enertech_Ai2020(sto):
    name, (x, y) = graphite_ocp_Enertech_Ai2020_data
    return pybamm.Interpolant(x, y, sto, name=name, interpolator="cubic")


# Call dict via a function to avoid errors when editing in place
def get_parameter_values():
    """
    Parameters for a composite graphite/silicon negative electrode, from the paper
    :footcite:t:`Ai2022`, based on the paper :footcite:t:`Chen2020`, and references
    therein.

    SEI parameters are example parameters for composite SEI on silicon/graphite. Both
    phases use the same values, from the paper :footcite:t:`Yang2017`
    """

    return {
        "chemistry": "lithium_ion",
        # sei
        "Primary: Ratio of lithium moles to SEI moles": 2.0,
        "Primary: SEI partial molar volume [m3.mol-1]": 9.585e-05,
        "Primary: SEI reaction exchange current density [A.m-2]": 1.5e-07,
        "Primary: SEI resistivity [Ohm.m]": 200000.0,
        "Primary: SEI solvent diffusivity [m2.s-1]": 2.5e-22,
        "Primary: Bulk solvent concentration [mol.m-3]": 2636.0,
        "Primary: SEI open-circuit potential [V]": 0.4,
        "Primary: SEI electron conductivity [S.m-1]": 8.95e-14,
        "Primary: SEI lithium interstitial diffusivity [m2.s-1]": 1e-20,
        "Primary: Lithium interstitial reference concentration [mol.m-3]": 15.0,
        "Primary: Initial SEI thickness [m]": 5e-09,
        "Primary: EC initial concentration in electrolyte [mol.m-3]": 4541.0,
        "Primary: EC diffusivity [m2.s-1]": 2e-18,
        "Primary: SEI kinetic rate constant [m.s-1]": 1e-12,
        "Primary: SEI growth activation energy [J.mol-1]": 0.0,
        "Secondary: Ratio of lithium moles to SEI moles": 2.0,
        "Secondary: SEI partial molar volume [m3.mol-1]": 9.585e-05,
        "Secondary: SEI reaction exchange current density [A.m-2]": 1.5e-07,
        "Secondary: SEI resistivity [Ohm.m]": 200000.0,
        "Secondary: SEI solvent diffusivity [m2.s-1]": 2.5e-22,
        "Secondary: Bulk solvent concentration [mol.m-3]": 2636.0,
        "Secondary: SEI open-circuit potential [V]": 0.4,
        "Secondary: SEI electron conductivity [S.m-1]": 8.95e-14,
        "Secondary: SEI lithium interstitial diffusivity [m2.s-1]": 1e-20,
        "Secondary: Lithium interstitial reference concentration [mol.m-3]": 15.0,
        "Secondary: Initial SEI thickness [m]": 5e-09,
        "Secondary: EC initial concentration in electrolyte [mol.m-3]": 4541.0,
        "Secondary: EC diffusivity [m2.s-1]": 2e-18,
        "Secondary: SEI kinetic rate constant [m.s-1]": 1e-12,
        "Secondary: SEI growth activation energy [J.mol-1]": 0.0,
        "Positive electrode reaction-driven LAM factor [m3.mol-1]": 0.0,
        # Plating parameters referred from OKane2022
        "Lithium metal partial molar volume [m3.mol-1]": 1.3e-05,
        "Primary: Lithium plating kinetic rate constant [m.s-1]": 1e-09,
        "Primary: Exchange-current density for plating [A.m-2]"
        "": graphite_plating_exchange_current_density_OKane2020,
        "Primary: Exchange-current density for stripping [A.m-2]"
        "": graphite_stripping_exchange_current_density_OKane2020,
        "Primary: Initial plated lithium concentration [mol.m-3]": 0.0,
        "Primary: Typical plated lithium concentration [mol.m-3]": 1000.0,
        "Primary: Lithium plating transfer coefficient": 0.65,
        "Primary: Dead lithium decay constant [s-1]": 1e-06,
        "Primary: Dead lithium decay rate [s-1]"
        "": graphite_SEI_limited_dead_lithium_OKane2022,
        "Secondary: Lithium plating kinetic rate constant [m.s-1]": 1e-09,
        "Secondary: Exchange-current density for plating [A.m-2]"
        "": silicon_plating_exchange_current_density_OKane2020,
        "Secondary: Exchange-current density for stripping [A.m-2]"
        "": silicon_stripping_exchange_current_density_OKane2020,
        "Secondary: Initial plated lithium concentration [mol.m-3]": 0.0,
        "Secondary: Typical plated lithium concentration [mol.m-3]": 1000.0,
        "Secondary: Lithium plating transfer coefficient": 0.65,
        "Secondary: Dead lithium decay constant [s-1]": 1e-06,
        "Secondary: Dead lithium decay rate [s-1]"
        "": silicon_SEI_limited_dead_lithium_OKane2022,
        # cell
        "Negative current collector thickness [m]": 1.2e-05,
        "Negative electrode thickness [m]": 8.52e-05,
        "Separator thickness [m]": 1.2e-05,
        "Positive electrode thickness [m]": 7.56e-05,
        "Positive current collector thickness [m]": 1.6e-05,
        "Electrode height [m]": 0.065,
        "Electrode width [m]": 1.58,
        "Cell cooling surface area [m2]": 0.00531,
        "Cell volume [m3]": 2.42e-05,
        "Cell thermal expansion coefficient [m.K-1]": 1.1e-06,
        "Negative current collector conductivity [S.m-1]": 58411000.0,
        "Positive current collector conductivity [S.m-1]": 36914000.0,
        "Negative current collector density [kg.m-3]": 8960.0,
        "Positive current collector density [kg.m-3]": 2700.0,
        "Negative current collector specific heat capacity [J.kg-1.K-1]": 385.0,
        "Positive current collector specific heat capacity [J.kg-1.K-1]": 897.0,
        "Negative current collector thermal conductivity [W.m-1.K-1]": 401.0,
        "Positive current collector thermal conductivity [W.m-1.K-1]": 237.0,
        "Nominal cell capacity [A.h]": 5.0,
        "Current function [A]": 5.0,
        "Contact resistance [Ohm]": 0,
        # negative electrode
        "Negative electrode conductivity [S.m-1]": 215.0,
        "Primary: Maximum concentration in negative electrode [mol.m-3]": 28700.0,
        "Primary: Initial concentration in negative electrode [mol.m-3]": 27700.0,
        "Primary: Negative particle diffusivity [m2.s-1]": graphite_LGM50_diffusivity_Chen2020,
        "Primary: Negative particle diffusivity constant [m2.s-1]": 5.5e-14,
        "Primary: Negative particle diffusivity activation energy [J.mol-1]": 30300.0,
        "Primary: Negative electrode OCP [V]": graphite_ocp_Enertech_Ai2020,
        "Negative electrode porosity": 0.25,
        "Primary: Negative electrode active material volume fraction": 0.735,
        "Primary: Negative particle radius [m]": 5.86e-06,
        "Negative electrode Bruggeman coefficient (electrolyte)": 1.5,
        "Negative electrode Bruggeman coefficient (electrode)": 0,
        "Negative electrode charge transfer coefficient": 0.5,
        "Negative electrode double-layer capacity [F.m-2]": 0.2,
        "Primary: Negative electrode exchange-current density [A.m-2]"
        "": graphite_LGM50_electrolyte_exchange_current_density_Chen2020,
        "Primary: Negative electrode kinetic rate constant [A.m-2]": 6.48e-7,
        "Primary: Negative electrode exchange-current density activation energy [J.mol-1]": 35000.0,
        "Primary: Negative electrode density [kg.m-3]": 1657.0,
        "Negative electrode specific heat capacity [J.kg-1.K-1]": 700.0,
        "Negative electrode thermal conductivity [W.m-1.K-1]": 1.7,
        "Primary: Negative electrode OCP entropic change [V.K-1]": 0.0,
        "Secondary: Maximum concentration in negative electrode [mol.m-3]": 278000.0,
        "Secondary: Initial concentration in negative electrode [mol.m-3]": 276610.0,
        "Secondary: Negative particle diffusivity [m2.s-1]": silicon_LGM50_diffusivity_Chen2020,
        "Secondary: Negative particle diffusivity constant [m2.s-1]": 1.67e-14,
        "Secondary: Negative particle diffusivity activation energy [J.mol-1]": 30300.0,
        "Secondary: Negative electrode lithiation OCP [V]"
        "": silicon_ocp_lithiation_Mark2016,
        "Secondary: Negative electrode delithiation OCP [V]"
        "": silicon_ocp_delithiation_Mark2016,
        "Secondary: Negative electrode OCP [V]": silicon_ocp_average_Mark2016,
        "Secondary: Negative electrode active material volume fraction": 0.015,
        "Secondary: Negative particle radius [m]": 1.52e-06,
        "Secondary: Negative electrode exchange-current density [A.m-2]"
        "": silicon_LGM50_electrolyte_exchange_current_density_Chen2020,
        "Secondary: Negative electrode kinetic rate constant [A.m-2]": 6.48e-7 * 28700 / 278000,
        "Secondary: Negative electrode exchange-current density activation energy [J.mol-1]": 35000.0,
        "Secondary: Negative electrode density [kg.m-3]": 2650.0,
        "Secondary: Negative electrode OCP entropic change [V.K-1]": 0.0,
        #cracking
        "Primary: Initial SEI on cracks thickness [m]": 5e-13,
        "Primary: Negative electrode Poisson's ratio": 0.3,
        "Primary: Negative electrode Young's modulus [Pa]": 15000000000.0,
        "Primary: Negative electrode reference concentration for free of deformation [mol.m-3]"
        "": 0.0,
        "Primary: Negative electrode partial molar volume [m3.mol-1]": 3.1e-06,
        "Primary: Negative electrode volume change": graphite_volume_change_Ai2020,
        "Primary: Negative electrode initial crack length [m]": 2e-08,
        "Primary: Negative electrode initial crack width [m]": 1.5e-08,
        "Primary: Negative electrode number of cracks per unit area [m-2]": 3180000000000000.0,
        "Primary: Negative electrode Paris' law constant b": 1.12,
        "Primary: Negative electrode Paris' law constant m": 2.2,
        "Primary: Negative electrode cracking rate": graphite_cracking_rate_Ai2020,
        "Primary: Negative electrode cracking rate constant [m/(Pa.m0.5)^m_cr]": 3.9e-20,
        "Primary: Negative electrode LAM constant proportional term [s-1]": 2.7778e-07,
        "Primary: Negative electrode LAM constant exponential term": 2.0,
        "Primary: Negative electrode critical stress [Pa]": 60000000.0,
        "Secondary: Initial SEI on cracks thickness [m]": 5e-13,
        "Secondary: Negative electrode Poisson's ratio": 0.3,
        "Secondary: Negative electrode Young's modulus [Pa]": 15000000000.0,
        "Secondary: Negative electrode reference concentration for free of deformation [mol.m-3]"
        "": 0.0,
        "Secondary: Negative electrode partial molar volume [m3.mol-1]": 3.1e-06,
        "Secondary: Negative electrode volume change": graphite_volume_change_Ai2020,
        "Secondary: Negative electrode initial crack length [m]": 2e-08,
        "Secondary: Negative electrode initial crack width [m]": 1.5e-08,
        "Secondary: Negative electrode number of cracks per unit area [m-2]": 3180000000000000.0,
        "Secondary: Negative electrode Paris' law constant b": 1.12,
        "Secondary: Negative electrode Paris' law constant m": 2.2,
        "Secondary: Negative electrode cracking rate": graphite_cracking_rate_Ai2020,
        "Secondary: Negative electrode cracking rate constant [m/(Pa.m0.5)^m_cr]": 3.9e-20,
        "Secondary: Negative electrode LAM constant proportional term [s-1]": 2.7778e-07,
        "Secondary: Negative electrode LAM constant exponential term": 2.0,
        "Secondary: Negative electrode critical stress [Pa]": 60000000.0,
        "Negative electrode Poisson's ratio": 0.3,
        "Negative electrode Young's modulus [Pa]": 15000000000.0,
        "Negative electrode reference concentration for free of deformation [mol.m-3]"
        "": 0.0,
        "Negative electrode partial molar volume [m3.mol-1]": 3.1e-06,
        "Negative electrode volume change": graphite_volume_change_Ai2020,
        "Negative electrode initial crack length [m]": 2e-08,
        "Negative electrode initial crack width [m]": 1.5e-08,
        "Negative electrode number of cracks per unit area [m-2]": 3180000000000000.0,
        "Negative electrode Paris' law constant b": 1.12,
        "Negative electrode Paris' law constant m": 2.2,
        "Negative electrode cracking rate": graphite_cracking_rate_Ai2020,
        "Negative electrode cracking rate constant [m/(Pa.m0.5)^m_cr]": 3.9e-20,
        "Negative electrode LAM constant proportional term [s-1]": 2.7778e-07,
        "Negative electrode LAM constant exponential term": 2.0,
        "Negative electrode critical stress [Pa]": 60000000.0,
        "Positive electrode Poisson's ratio": 0.2,
        "Positive electrode Young's modulus [Pa]": 375000000000.0,
        "Positive electrode reference concentration for free of deformation [mol.m-3]"
        "": 0.0,
        "Positive electrode partial molar volume [m3.mol-1]": 1.25e-05,
        "Positive electrode volume change": volume_change_Ai2020,
        "Positive electrode initial crack length [m]": 2e-08,
        "Positive electrode initial crack width [m]": 1.5e-08,
        "Positive electrode number of cracks per unit area [m-2]": 3180000000000000.0,
        "Positive electrode Paris' law constant b": 1.12,
        "Positive electrode Paris' law constant m": 2.2,
        "Positive electrode cracking rate": cracking_rate_Ai2020,
        "Positive electrode cracking rate constant [m/(Pa.m0.5)^m_cr]": 3.9e-20,
        "Positive electrode LAM constant proportional term [s-1]": 2.7778e-07,
        "Positive electrode LAM constant exponential term": 2.0,
        "Positive electrode critical stress [Pa]": 375000000.0,
        
        # positive electrode
        "Positive electrode conductivity [S.m-1]": 0.18,
        "Maximum concentration in positive electrode [mol.m-3]": 63104.0,
        "Positive particle diffusivity [m2.s-1]": nmc_LGM50_diffusivity_Chen2020,
        "Positive particle diffusivity constant [m2.s-1]": 4e-15,
        "Positive particle diffusivity activation energy [J.mol-1]": 25000.0,
        "Positive electrode OCP [V]": nmc_LGM50_ocp_Chen2020,
        "Positive electrode porosity": 0.335,
        "Positive electrode active material volume fraction": 0.665,
        "Positive particle radius [m]": 5.22e-06,
        "Positive electrode Bruggeman coefficient (electrolyte)": 1.5,
        "Positive electrode Bruggeman coefficient (electrode)": 0,
        "Positive electrode charge transfer coefficient": 0.5,
        "Positive electrode double-layer capacity [F.m-2]": 0.2,
        "Positive electrode exchange-current density [A.m-2]"
        "": nmc_LGM50_electrolyte_exchange_current_density_Chen2020,
        "Positive electrode kinetic rate constant [A.m-2]": 3.42e-6,
        "Positive electrode exchange-current density activation energy [J.mol-1]": 17800.0,
        "Positive electrode density [kg.m-3]": 3262.0,
        "Positive electrode specific heat capacity [J.kg-1.K-1]": 700.0,
        "Positive electrode thermal conductivity [W.m-1.K-1]": 2.1,
        "Positive electrode OCP entropic change [V.K-1]": 0.0,
        # separator
        "Separator porosity": 0.47,
        "Separator Bruggeman coefficient (electrolyte)": 1.5,
        "Separator density [kg.m-3]": 397.0,
        "Separator specific heat capacity [J.kg-1.K-1]": 700.0,
        "Separator thermal conductivity [W.m-1.K-1]": 0.16,
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
        "Lower voltage cut-off [V]": 2.5,
        "Upper voltage cut-off [V]": 4.2,
        "Open-circuit voltage at 0% SOC [V]": 2.5,
        "Open-circuit voltage at 100% SOC [V]": 4.2,
        "Initial concentration in negative electrode [mol.m-3]": 29866.0,
        "Initial concentration in positive electrode [mol.m-3]": 17038.0,
        "Initial temperature [K]": 298.15,
        # citations
        "citations": ["Chen2020", "Ai2022"],
    }