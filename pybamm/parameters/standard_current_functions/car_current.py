#
# Car current (piecewise constant)
#


def car_current(t):
    """
    Piecewise constant current as a function of time in seconds. This is adapted
    from the file getCarCurrent.m, which is part of the LIONSIMBA toolbox [1]_.

    References
    ----------
    .. [1] M Torchio, L Magni, R Bushan Gopaluni, RD Braatz, and D. Raimondoa.
           LIONSIMBA: A Matlab framework based on a finite volume model suitable
           for Li-ion battery design, simulation, and control. Journal of The
           Electrochemical Society, 163(7):1192-1205, 2016.
    """

    current = (
        1 * (t >= 0) * (t <= 50)
        - 0.5 * (t > 50) * (t <= 60)
        + 0.5 * (t > 60) * (t <= 210)
        + 1 * (t > 210) * (t <= 410)
        + 2 * (t > 410) * (t <= 415)
        + 1.25 * (t > 415) * (t <= 615)
        - 0.5 * (t > 615)
    )

    return current
