def lfp_cond(c):
    """
    Conductivity of LiPF6 in EC:DMC as in Newman's Dualfoil code. This
    function is in dimensional form.

    Parameters
    ----------
        c: double
        lithium-ion concentration

    """
    c = c / 1000
    sigma_e = 0.0911 + 1.9101 * c - 1.052 * c ** 2 + 0.1554 * c ** 3
    return sigma_e
