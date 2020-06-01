from pybamm import tanh


def nco_ocp_Ecker2015_function(sto):
    """
    NCO OCP as a function of stochiometry [1, 2, 3].

    References
    ----------
    .. [1] Ecker, Madeleine, et al. "Parameterization of a physico-chemical model of
    a lithium-ion battery i. determination of parameters." Journal of the
    Electrochemical Society 162.9 (2015): A1836-A1848.
    .. [2] Ecker, Madeleine, et al. "Parameterization of a physico-chemical model of
    a lithium-ion battery ii. model validation." Journal of The Electrochemical
    Society 162.9 (2015): A1849-A1857.
    .. [3] Richardson, Giles, et. al. "Generalised single particle models for
    high-rate operation of graded lithium-ion electrodes: Systematic derivation
    and validation." Electrochemica Acta 339 (2020): 135862

    Parameters
    ----------
    sto : :class:`pybamm.Symbol`
       Stochiometry of material (li-fraction)

    """

    # LiNiCo from Ecker, Kabitz, Laresgoiti et al.
    # Analytical fit (WebPlotDigitizer + gnuplot)
    a = -2.35211
    c = 0.0747061
    d = 31.886
    e = 0.0219921
    g = 0.640243
    h = 5.48623
    i = 0.439245
    j = 3.82383
    k = 4.12167
    m = 0.176187
    n = 0.0542123
    o = 18.2919
    p = 0.762272
    q = 4.23285
    r = -6.34984
    s = 2.66395
    t = 0.174352

    u_eq = (
        a * sto
        - c * tanh(d * (sto - e))
        - r * tanh(s * (sto - t))
        - g * tanh(h * (sto - i))
        - j * tanh(k * (sto - m))
        - n * tanh(o * (sto - p))
        + q
    )
    return u_eq
