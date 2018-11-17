#
# Scales for nondimensionalisation
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import pybamm

class Scales(object):
    """Scales for non-dimensionalisation.

    Parameters
    ----------
    param : pybamm.Parameters() instance
        The parameters from which to calculate scales.

    """

    def __init__(self, param):
        # Length scale [m]
        self.length = param.L
        # Discharge time scale [h]
        self.time = param.cmax * param.F * param.L / param.ibar / 3600
        # Concentration scale [mol.m-3]
        self.conc = param.cmax
        # Current density scale [A.m-2]
        self.current = param.ibar
        # Interfacial current density scale (neg) [A.m-2]
        self.jn = param.ibar / (param.Anmax * param.L)
        # Interfacial current density scale (pos) [A.m-2]
        self.jp = param.ibar / (param.Apmax * param.L)
        # Interfacial area scale (neg) [m2.m-3]
        self.An = param.Anmax
        # Interfacial area scale (pos) [m2.m-3]
        self.Ap = param.Apmax
        # Interfacial area times current density [A.m-3]
        self.Aj = param.ibar / (param.L)
        # Voltage scale (thermal voltage) [V]
        self.pot = param.R * param.T_ref / param.F
        # Porosity, SOC scale [-]
        self.one = 1
        # Reaction velocity [m.s-1]
        self.U_rxn = param.ibar / (param.cmax * param.F)
        # Temperature scale [K]
        self.temp = param.T_max - param.T_inf

        # Combined scales
        self.It = self.current * self.time

        # Dictionary matching solution attributes
        # to re-dimensionalisation scales
        self.match = {
            "t": "time",
            "x": "length",
            "icell": "current",
            "Icircuit": "current",
            "intI": "It",
            "c0_v": "conc",
            "c1": "conc",
            "c": "conc",
            "c_avg": "conc",
            "cO2": "concO2",
            "cO2_avg": "concO2",
            "phi": ("pot", -param.U_Pb_ref),
            "phis": ("pot", 0, param.U_PbO2_ref - param.U_Pb_ref),
            "phisn": "pot",
            "phisp": ("pot", param.U_PbO2_ref - param.U_Pb_ref),
            "xi": ("pot", -param.U_Pb_ref, param.U_PbO2_ref),
            "xin": ("pot", -param.U_Pb_ref),
            "xis": ("one", 0),
            "xip": ("pot", param.U_PbO2_ref),
            "V0": ("pot", param.U_PbO2_ref - param.U_Pb_ref),
            "V1": ("pot", param.U_PbO2_ref - param.U_Pb_ref),
            "V": ("pot", param.U_PbO2_ref - param.U_Pb_ref),
            "V0circuit": ("pot", 6 * (param.U_PbO2_ref - param.U_Pb_ref)),
            "Vcircuit": ("pot", 6 * (param.U_PbO2_ref - param.U_Pb_ref)),
            "j": ("jn", "jp"),
            "jO2": ("jn", "jp"),
            "jH2": ("jn", "jp"),
            "jn": "jn",
            "js": "one",
            "jp": "jp",
            "jO2n": "jn",
            "jO2s": "one",
            "jO2p": "jp",
            "jH2n": "jn",
            "jH2s": "one",
            "jH2p": "jp",
            "A": ("An", "Ap"),
            "AO2": ("An", "Ap"),
            "AH2": ("An", "Ap"),
            "Adl": ("An", "Ap"),
            "An": "An",
            "As": "one",
            "Ap": "Ap",
            "AO2n": "An",
            "AO2s": "one",
            "AO2p": "Ap",
            "AH2n": "An",
            "AH2s": "one",
            "AH2p": "Ap",
            "Adln": "An",
            "Adls": "one",
            "Adlp": "Ap",
            "i": "current",
            "i_n": "current",
            "i_p": "current",
            "isolid": "current",
            "q": "one",
            "U": "one",
            "Un": "one",
            "Us": "one",
            "Up": "one",
            "T": ("temp", param.T_inf),
        }
