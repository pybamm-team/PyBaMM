#
# Dimensional and dimensionless parameters, and scales
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import pybamm

import pandas as pd
import numpy as np
import os

KNOWN_TESTS = ["", "convergence"]


def read_parameters_csv(chemistry, filename):
    """Reads parameters from csv file into dict.

    Parameters
    ----------
    chemistry : string
        The chemistry to read parameters for (name of folder)
    filename : string
        The name of the csv file containing the parameters.
        Must be a file in `input/parameters/`

    Returns
    -------
    dict
        {name: value} pairs for the parameters.

    """
    # Hack to access input/parameters from any working directory
    filename = os.path.join(
        pybamm.ABSOLUTE_PATH, "input", "parameters", chemistry, filename
    )

    #
    df = pd.read_csv(filename, comment="#", skip_blank_lines=True)
    # Drop rows that are all NaN (seems to not work with skip_blank_lines)
    df.dropna(how="all", inplace=True)
    return {k: v for (k, v) in zip(df.Name, df.Value)}


class Parameters(object):
    """
    The parameters for the simulation.

    Parameters
    ----------
    chemistry : string
        The chemistry to define parameters for
    current : dict, optional
        {"Ibar": float or int, "type": string}, defines the external current
    optional_parameters : dict or string, optional
        dict or string (calls csv file) of optional parameters to overwrite
        some of the default parameters
    tests : string, optional
        An option to change the parameters for easier testing:
            * '' (default) : no tests, normal operation
            * 'convergence' : convergence tests, simplify parameters
    """

    def __init__(self, tests=""):
        self._raw = {}

        # Tests
        if tests not in KNOWN_TESTS:
            raise NotImplementedError(
                """Tests '{}' are not implemented.
                   Valid choices: one of '{}'.""".format(
                    tests, KNOWN_TESTS
                )
            )
        self.tests = tests

    def update_raw(
        self,
        chemistry="lead-acid",
        base_parameters_file="default.csv",
        optional_parameters={},
        current=None,
    ):
        # Defaults
        # Load default parameters from csv file
        base_parameters = read_parameters_csv(chemistry, base_parameters_file)
        # Assign either default values
        self._raw.update(base_parameters)

        # Optional parameters
        # If optional_parameters is a filename, load from that filename
        if isinstance(optional_parameters, str):
            optional_parameters = read_parameters_csv(chemistry, optional_parameters)
        else:
            # Otherwise, optional_parameters should be a dict
            assert isinstance(
                optional_parameters, dict
            ), """optional_parameters should be a filename (string) or a dict,
                but it is a '{}'""".format(
                type(optional_parameters)
            )
        # Overwrite raw parameters with optional values where given
        self._raw.update(optional_parameters)

        # Input current
        # Set default
        if current is None:
            current = {"Ibar": 1, "type": "constant"}
        self.current = current

    @property
    def geometric(self):
        """
        Geometric parameters.
        *Chemistries*: lithium-ion, lead-acid
        """
        # Total width [m]
        L = self._raw["Ln"] + self._raw["Ls"] + self._raw["Lp"]
        # Area of the current collectors [m2]
        A_cc = self._raw["H"] * self._raw["W"]
        # Volume of a cell [m3]
        Vc = A_cc * L

        # Dimensionless half-width of negative electrode
        ln = self._raw["Ln"] / L
        # Dimensionless width of separator
        ls = self._raw["Ls"] / L
        # Dimensionless half-width of positive electrode
        lp = self._raw["Lp"] / L
        # Aspect ratio
        delta = L / self._raw["H"]

        # Return dict of all locally defined variables except self
        return {key: value for key, value in locals().items() if key != "self"}

    @property
    def electrical(self):
        """
        Parameters relating to the external electrical circuit
        *Chemistries*: lithium-ion, lead-acid
        """
        # Reference current density [A.m-2]
        ibar = abs(self.current["Ibar"]) / (
            self._raw["n_electrodes_parallel"] * self.geometric["A_cc"]
        )
        # C-rate [-]
        Crate = self.current["Ibar"] / self._raw["Q"]

        # Dimensionless voltage cut-off
        voltage_cutoff = self.scales["pot"] * (
            self._raw["voltage_cutoff_circuit"] / self._raw["n_cells_series"]
            - (self._raw["U_PbO2_ref"] - self._raw["U_Pb_ref"])
        )
        # Return dict of all locally defined variables except self
        return {key: value for key, value in locals().items() if key != "self"}

    @property
    def electrolyte(self):
        """
        Parameters for the electrolyte
        *Chemistries*: lithium-ion, lead-acid
        """
        # Effective reaction rates
        # Main reaction (neg) [-]
        sn = -(self._raw["spn"] + 2 * self._raw["tpw"]) / 2
        # Main reaction (pos) [-]
        sp = -(self._raw["spp"] + 2 * self._raw["tpw"]) / 2

        # Diffusional C-rate: diffusion timescale/discharge timescale
        Cd = (
            (self.geometric["L"] ** 2)
            / self.D_hat(self._raw["cmax"])
            / (
                self._raw["cmax"]
                * self._raw["F"]
                * self.geometric["L"]
                / self.electrical["ibar"]
            )
        )

        # Return dict of all locally defined variables except self
        return {key: value for key, value in locals().items() if key != "self"}

    @property
    def neg_electrode(self):
        """
        Negative electrode physical parameters
        *Chemistries*: lithium-ion, lead-acid
        """
        # Effective lead conductivity (Bruggeman) [S.m-1]
        sigma_eff = self._raw["sigma_n"] * (1 - self._raw["epsnmax"]) ** 1.5

        # Dimensionless lead conductivity
        iota_s = (
            sigma_eff
            * self.scales["pot"]
            / (self.geometric["L"] * self.electrical["ibar"])
        )
        # Dimensionless electrode capacity (neg)
        Qmax = self._raw["Qnmax_hat"] / (self._raw["cmax"] * self._raw["F"])

        # Return dict of all locally defined variables except self
        return {key: value for key, value in locals().items() if key != "self"}

    @property
    def pos_electrode(self):
        """
        Positive electrode physical parameters
        *Chemistries*: lithium-ion, lead-acid
        """
        # Effective lead dioxide conductivity (Bruggeman) [S.m-1]
        sigma_eff = self._raw["sigma_p"] * (1 - self._raw["epspmax"]) ** 1.5

        # Dimensionless lead dioxide conductivity
        iota_s = (
            sigma_eff
            * self.scales["pot"]
            / (self.geometric["L"] * self.electrical["ibar"])
        )
        # Dimensionless electrode capacity (pos)
        Qmax = self._raw["Qpmax_hat"] / (self._raw["cmax"] * self._raw["F"])

        # Return dict of all locally defined variables except self
        return {key: value for key, value in locals().items() if key != "self"}

    @property
    def neg_reactions(self):
        """
        Parameters for reactions in the negative electrode
        *Chemistries*: lithium-ion, lead-acid
        """
        # Dimensionless exchange-current density (neg)
        iota_ref = self._raw["jref_n"] / self.scales["jn"]
        # Dimensionless double-layer capacity (neg)
        gamma_dl = (
            self._raw["Cdl"]
            * self.scales["pot"]
            / self.scales["jn"]
            / (self.scales["time"] * 3600)
        )

        # Return dict of all locally defined variables except self
        return {key: value for key, value in locals().items() if key != "self"}

    @property
    def pos_reactions(self):
        """
        Parameters for reactions in the positive electrode
        *Chemistries*: lithium-ion, lead-acid
        """
        # Dimensionless exchange-current density (pos)
        iota_ref = self._raw["jref_p"] / self.scales["jp"]
        # Dimensionless double-layer capacity (pos)
        gamma_dl = (
            self._raw["Cdl"]
            * self.scales["pot"]
            / self.scales["jp"]
            / (self.scales["time"] * 3600)
        )

        # Return dict of all locally defined variables except self
        return {key: value for key, value in locals().items() if key != "self"}

    @property
    def neg_volume_changes(self):
        """
        Parameters for volume changes in the negative electrode
        *Chemistries*: lead-acid
        """
        # Net Molar Volume consumed in neg electrode [m3.mol-1]
        DeltaVsurf = self._raw["VPbSO4"] - self._raw["VPb"]

        # Dimensionless molar volume change (lead)
        beta_surf = self._raw["cmax"] * DeltaVsurf / 2

        # Return dict of all locally defined variables except self
        return {key: value for key, value in locals().items() if key != "self"}

    @property
    def pos_volume_changes(self):
        """
        Parameters for volume changes in the positive electrode
        *Chemistries*: lead-acid
        """
        # Net Molar Volume consumed in pos electrode [m3.mol-1]
        DeltaVsurf = self._raw["VPbO2"] - self._raw["VPbSO4"]

        # Dimensionless molar volume change (lead dioxide)
        beta_surf = self._raw["cmax"] * DeltaVsurf / 2

        # Return dict of all locally defined variables except self
        return {key: value for key, value in locals().items() if key != "self"}

    @property
    def temperature(self):
        """
        Temperature parameters
        *Chemistries*: lithium-ion, lead-acid
        """
        # External temperature [K]
        self.T_inf = self._raw["T_ref"]

        # Return dict of all locally defined variables except self
        return {key: value for key, value in locals().items() if key != "self"}

    @property
    def lead_acid_misc(self):
        """Miscellaneous parameters for lead-acid"""
        # Excluded volume fraction
        alpha = (2 * self._raw["Vw"] - self._raw["Ve"]) * self.scales["conc"]
        # Ratio of viscous pressure scale to osmotic pressure scale
        pi_os = (
            self.mu_hat(self._raw["cmax"])
            * self.scales["U_rxn"]
            * self.geometric["L"]
            / (
                self._raw["d"] ** 2
                * self._raw["R"]
                * self._raw["T_ref"]
                * self._raw["cmax"]
            )
        )

        # Return dict of all locally defined variables except self
        return {key: value for key, value in locals().items() if key != "self"}

    def initial_conditions(self):
        ################################################################################
        # Initial conditions (dimensionless) ###########################################
        # Concentration
        self.c0 = self.q0
        # Dimensionless max capacity
        self.qmax = (
            (self.Ln * self.epsnmax + self.Ls * self.epssmax + self.Lp * self.epspmax)
            / self.L
            / (self.sp - self.sn)
        )
        # Initial electrode states of charge
        self.Un0 = self.qmax / (self.Qnmax * self.ln) * (1 - self.q0)
        self.Up0 = self.qmax / (self.Qpmax * self.lp) * (1 - self.q0)
        # Initial porosities
        self.epsDeltan = self.beta_surf_n / self.ln * self.qmax
        self.epsDeltap = self.beta_surf_p / self.lp * self.qmax
        # Negative electrode [-]
        self.epsln0 = self.epsnmax - self.epsDeltan * (1 - self.q0)
        # Separator [-]
        self.epsls0 = self.epssmax
        # Positive electrode [-]
        self.epslp0 = self.epspmax - self.epsDeltap * (1 - self.q0)
        ################################################################################
        ################################################################################

    def set_mesh_dependent_parameters(self, mesh):
        """Create parameters that depend on the mesh
        (e.g. different in each electrode and separator).

        Parameters
        ----------
        mesh : pybamm.mesh.Mesh() instance
            The mesh on which to evaluate the parameters.

        """
        if self.tests == "":
            self.s = np.concatenate(
                [
                    self.sn * np.ones_like(mesh.xcn),
                    np.zeros_like(mesh.xcs),
                    self.sp * np.ones_like(mesh.xcp),
                ]
            )
        elif self.tests == "convergence":
            # Set s=1 everywhere so we don't need to worry about source terms
            self.s = np.ones_like(mesh.xc)

    def Icircuit(self, t):
        """The current in the external circuit.

        Parameters
        ----------
        t : float or array_like, shape (n,)
            Time in *hours*.

        Returns
        -------
        array_like, shape () or array_like, shape (n,)
            The current at time(s) t, in Amps.

        """
        if self.current["type"] == "constant":
            return self.current["Ibar"] * np.ones_like(t)

    def icell(self, t):
        """The dimensionless current function (could be some data)"""
        # This is a function of dimensionless time; Icircuit is a function of
        # time in *hours*
        return (
            self.Icircuit(t * self.scales["time"])
            / (self._raw["n_electrodes_parallel"] * self.geometric["A_cc"])
            / self.electrical["ibar"]
        )

    @property
    def scales(self):
        # Length scale [m]
        length = self.geometric["L"]
        # Discharge time scale [h]
        time = (
            self._raw["cmax"]
            * self._raw["F"]
            * self.geometric["L"]
            / self.electrical["ibar"]
            / 3600
        )
        # Concentration scale [mol.m-3]
        conc = self._raw["cmax"]
        # Current density scale [A.m-2]
        current = self.electrical["ibar"]
        # Interfacial current density scale (neg) [A.m-2]
        jn = self.electrical["ibar"] / (self._raw["Anmax"] * self.geometric["L"])
        # Interfacial current density scale (pos) [A.m-2]
        jp = self.electrical["ibar"] / (self._raw["Apmax"] * self.geometric["L"])
        # Interfacial area scale (neg) [m2.m-3]
        An = self._raw["Anmax"]
        # Interfacial area scale (pos) [m2.m-3]
        Ap = self._raw["Apmax"]
        # Interfacial area times current density [A.m-3]
        Aj = self.electrical["ibar"] / (self.geometric["L"])
        # Voltage scale (thermal voltage) [V]
        pot = self._raw["R"] * self._raw["T_ref"] / self._raw["F"]
        # Porosity, SOC scale [-]
        one = 1
        # Reaction velocity [m.s-1]
        U_rxn = self.electrical["ibar"] / (self._raw["cmax"] * self._raw["F"])
        # Temperature scale [K]
        temp = self._raw["T_max"] - self._raw["T_inf"]

        # Combined scales
        It = current * time

        # Dictionary matching solution attributes to re-dimensionalisation scales
        match = {
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

        # Return dict of all locally defined variables except self
        return {key: value for key, value in locals().items() if key != "self"}
