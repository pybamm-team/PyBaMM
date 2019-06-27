#
# Car current (piecewise constant)
#
import numpy as np


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
    current = np.piecewise(
        t,
        [
            (t >= 0) & (t <= 50),
            (t > 60) & (t <= 210),
            (t > 210) & (t <= 410),
            (t > 410) & (t <= 415),
            (t > 415) & (t <= 615),
            (t > 615),
        ],
        [1, -0.5, 0.5, 1, 2, 1.25, -0.5],
    )
    return current
