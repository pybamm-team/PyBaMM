#
# Experiment class
#

import numpy as np
import re

class CC:
    """
    Class for constant current experimental steps conditions under which to run the model. In general, a
    list of operating conditions should be passed in. Each operating condition should
    be of the form "C rate = 1" or "duration = 3600 s".

    Parameters
    ----------
    C_rate : float
        C rate for charging
    temperature : float, optional
        The ambient air temperature in degrees Celsius at which to run the experiment.
        Default is None whereby the ambient temperature is taken from the parameter set.
        This value is overwritten if the temperature is specified in a step.
    duration : float, optional
        The time period in seconds
    upper_cutoff : float, optional
        The upper cutoff voltage which is 4.2V here.
    """
    def __init__(
            self,
            C_rate="",
            temperature=None,
            duration=None,
            upper_cutoff=None,
                 ):
        self.c_rate = C_rate
        self.temperature = temperature
        self.duration = duration
        self.upper_cutoff = upper_cutoff

class CV:
    """
    Class for constant voltage experimental steps conditions under which to run the model.


    Parameters
    ----------
    voltage : float
        voltage value for constant voltage charge
    temperature : float, optional
        The ambient air temperature in degrees Celsius at which to run the experiment.
        Default is None whereby the ambient temperature is taken from the parameter set.
        This value is overwritten if the temperature is specified in a step.
    duration : float, optional
        The time period in seconds.
    lower_cutoff : float, optional
        The lower cutoff value of current which is 50mA here
    """
    def __init__(
            self,
            voltage="",
            temperature=None,
            duration=None,
            lower_cutoff=None,
                 ):
        self.voltage = voltage
        self.temperature = temperature
        self.duration = duration
        self.lower_cutoff = lower_cutoff

class Rest:
    """
    Class for rest period experimental steps conditions under which to run the model.


    Parameters
    ----------
    duration : float
        The time period in seconds.
    temperature : float, optional
        The ambient air temperature in degrees Celsius at which to run the experiment.
        Default is None whereby the ambient temperature is taken from the parameter set.
        This value is overwritten if the temperature is specified in a step.
    """
    def __init__(
            self,
            duration="",
            temperature=None,
                 ):
        self.duration = duration
        self.temperature = temperature



