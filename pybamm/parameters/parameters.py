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
KNOWN_CHEMISTRIES = ["lithium-ion", "lead-acid"]
CHEMISTRY_PROPERTIES = {
    "lead-acid": (
        "geometric",
        "electrical",
        "electrolyte",
        "neg_electrode",
        "pos_electrode",
        "neg_reactions",
        "pos_reactions",
        "neg_volume_changes",
        "pos_volume_changes",
        "temperature",
        "lead_acid_misc",
    )
}


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

    The aim of this class is to:
    - Be flexible enough to contain parameters from different chemistries, which will
        overlap but have some differences.
    - Write parameters in a way that submodels only access the parameters they need to.

    We achieve this by defining subparams that return subclasses of specific
    parameters. For each chemistry, submodels will only ever call subparams specific to
    themselves, and so never see parameters that aren't defined for their chemistry.
    Further, submodels that can be applied to either the negative electrode or the
    positive electrode will receive parameters in the same format in either case, which
    avoid clunky if statements in the submodel statements

    Parameters
    ----------
    chemistry : string
        The chemistry to define parameters for
    base_parameters_file : string
        The file from which to read the base parameters
    optional_parameters : dict or string, optional
        dict or string (calls csv file) of optional parameters to overwrite
        some of the default parameters
    current : dict, optional
        {"Ibar": float or int, "type": string}, defines the external current
    tests : string, optional
        An option to change the parameters for easier testing:
            * '' (default) : no tests, normal operation
            * 'convergence' : convergence tests, simplify parameters

    Examples
    --------
    >>> param = Parameters(chemistry="lead-acid")
    >>> param.geometric["ln"]
    0.24657534246575338
    >>> param.geometric["lp"]
    0.3424657534246575
    >>> param.neg_reactions["l"]
    0.24657534246575338
    >>> param.pos_reactions["l"]
    0.3424657534246575
    """

    def __init__(
        self,
        chemistry="lead-acid",
        current=None,
        tests="",
        base_parameters_file="default.csv",
        optional_parameters={},
    ):
        # Chemistry
        if chemistry not in KNOWN_CHEMISTRIES:
            raise NotImplementedError(
                """Chemistry '{}' is not implemented.
                   Valid choices: one of '{}'.""".format(
                    chemistry, KNOWN_CHEMISTRIES
                )
            )
        assert isinstance(CHEMISTRY_PROPERTIES[chemistry], tuple)
        self._chemistry = chemistry

        # Input current
        # Set default
        if current is None:
            current = {"Ibar": 1, "type": "constant"}
        self.current = current

        # Tests
        if tests not in KNOWN_TESTS:
            raise NotImplementedError(
                """Tests '{}' are not implemented.
                   Valid choices: one of '{}'.""".format(
                    tests, KNOWN_TESTS
                )
            )
        self.tests = tests

        # Functions from the correct module
        self._func = pybamm.__dict__["functions_" + chemistry.replace("-", "_")]

        # Default parameters
        # Load default parameters from csv file
        self._raw = read_parameters_csv(chemistry, base_parameters_file)

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

        # Initial with empty mesh
        self._mesh = None

        # Overwrite raw parameters with optional values where given
        # Simultaneously set subparams
        self.update_raw(optional_parameters)

    def set_subparams(self):
        # Unset subparams (to avoid cross-calling with old subparams)
        for subparam in CHEMISTRY_PROPERTIES[self._chemistry]:
            self.unset_subparam(subparam)
        # Set new subparams - order matters
        for subparam in CHEMISTRY_PROPERTIES[self._chemistry]:
            self.set_subparam(subparam)

    def update_raw(self, new_parameters):
        """
        Update raw parameter values with dict.

        Parameters
        ----------
        new_parameters : dict
            dict of optional parameters to overwrite some of the default parameters

        """
        # Update _raw dict
        self._raw.update(new_parameters)
        self.set_subparams()

    def set_mesh(self, mesh):
        """
        Set the mesh for parameters that depend on the mesh (e.g. parameters across the
        whole electrode sandwich).

        Parameters
        ----------
        mesh : :class:`pybamm.mesh.Mesh` instance
            The mesh for the parameters
        """
        self._mesh = mesh
        self.set_subparams()

    def unset_subparam(self, subparam):
        self.__dict__[subparam] = None

    def set_subparam(self, subparam):
        subparam_class_name = name_string_to_class_string(subparam)
        self.__dict__[subparam] = globals()[subparam_class_name](self)

    # def initial_conditions(self):
    #     ##############################################################################
    #     # Initial conditions (dimensionless) #########################################
    #     # Concentration
    #     self.c0 = self.q0
    #     # Dimensionless max capacity
    #     self.qmax = (
    #         (self.Ln * self.epsnmax + self.Ls * self.epssmax + self.Lp * self.epspmax)
    #         / self.L
    #         / (self.sp - self.sn)
    #     )
    #     # Initial electrode states of charge
    #     self.Un0 = self.qmax / (self.Qnmax * self.ln) * (1 - self.q0)
    #     self.Up0 = self.qmax / (self.Qpmax * self.lp) * (1 - self.q0)
    #     # Initial porosities
    #     self.epsDeltan = self.beta_surf_n / self.ln * self.qmax
    #     self.epsDeltap = self.beta_surf_p / self.lp * self.qmax
    #     # Negative electrode [-]
    #     self.epsln0 = self.epsnmax - self.epsDeltan * (1 - self.q0)
    #     # Separator [-]
    #     self.epsls0 = self.epssmax
    #     # Positive electrode [-]
    #     self.epslp0 = self.epspmax - self.epsDeltap * (1 - self.q0)
    #     ##############################################################################
    #     ##############################################################################

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
            / (self._raw["n_electrodes_parallel"] * self.geometric.A_cc)
            / self.electrical.ibar
        )

    @property
    def scales(self):
        # Length scale [m]
        length = self.geometric.L
        # Discharge time scale [h]
        time = (
            self._raw["cmax"]
            * self._raw["F"]
            * self.geometric.L
            / self.electrical.ibar
            / 3600
        )
        # Concentration scale [mol.m-3]
        conc = self._raw["cmax"]
        # Current density scale [A.m-2]
        current = self.electrical.ibar
        # Interfacial current density scale (neg) [A.m-2]
        jn = self.electrical.ibar / (self._raw["Anmax"] * self.geometric.L)
        # Interfacial current density scale (pos) [A.m-2]
        jp = self.electrical.ibar / (self._raw["Apmax"] * self.geometric.L)
        # Interfacial area scale (neg) [m2.m-3]
        An = self._raw["Anmax"]
        # Interfacial area scale (pos) [m2.m-3]
        Ap = self._raw["Apmax"]
        # Interfacial area times current density [A.m-3]
        Aj = self.electrical.ibar / (self.geometric.L)  # noqa: F841
        # Voltage scale (thermal voltage) [V]
        pot = self._raw["R"] * self._raw["T_ref"] / self._raw["F"]
        # Porosity, SOC scale [-]
        one = 1
        # Reaction velocity [m.s-1]
        U_rxn = self.electrical.ibar / (  # noqa:F841
            self._raw["cmax"] * self._raw["F"]
        )  # noqa: F841
        # Temperature scale [K]
        # temp = self._raw["T_max"] - self._raw["T_inf"]

        # Combined scales
        It = current * time

        # Dictionary matching solution attributes to re-dimensionalisation scales
        match = {  # noqa: F841
            "t": time,
            "x": length,
            "icell": current,
            "Icircuit": current,
            "intI": It,
            "c0_v": conc,
            "c1": conc,
            "c": conc,
            "c_avg": conc,
            # "cO2": concO2,
            # "cO2_avg": concO2,
            "phi": (pot, -self._raw["U_Pb_ref"]),
            "phis": (pot, 0, self._raw["U_PbO2_ref"] - self._raw["U_Pb_ref"]),
            "phisn": pot,
            "phisp": (pot, self._raw["U_PbO2_ref"] - self._raw["U_Pb_ref"]),
            "xi": (pot, -self._raw["U_Pb_ref"], self._raw["U_PbO2_ref"]),
            "xin": (pot, -self._raw["U_Pb_ref"]),
            "xis": (one, 0),
            "xip": (pot, self._raw["U_PbO2_ref"]),
            "V0": (pot, self._raw["U_PbO2_ref"] - self._raw["U_Pb_ref"]),
            "V1": (pot, self._raw["U_PbO2_ref"] - self._raw["U_Pb_ref"]),
            "V": (pot, self._raw["U_PbO2_ref"] - self._raw["U_Pb_ref"]),
            "V0circuit": (pot, 6 * (self._raw["U_PbO2_ref"] - self._raw["U_Pb_ref"])),
            "Vcircuit": (pot, 6 * (self._raw["U_PbO2_ref"] - self._raw["U_Pb_ref"])),
            "j": (jn, jp),
            "jO2": (jn, jp),
            "jH2": (jn, jp),
            "jn": jn,
            "js": one,
            "jp": jp,
            "jO2n": jn,
            "jO2s": one,
            "jO2p": jp,
            "jH2n": jn,
            "jH2s": one,
            "jH2p": jp,
            "A": (An, Ap),
            "AO2": (An, Ap),
            "AH2": (An, Ap),
            "Adl": (An, Ap),
            "An": An,
            "As": one,
            "Ap": Ap,
            "AO2n": An,
            "AO2s": one,
            "AO2p": Ap,
            "AH2n": An,
            "AH2s": one,
            "AH2p": Ap,
            "Adln": An,
            "Adls": one,
            "Adlp": Ap,
            "i": current,
            "i_n": current,
            "i_p": current,
            "isolid": current,
            "q": one,
            "U": one,
            "Un": one,
            "Us": one,
            "Up": one,
            # "T": (temp, self.temperature["T_inf"]),
        }

        # Return dict of all locally defined variables except self
        return {key: value for key, value in locals().items() if key != "self"}


def name_string_to_class_string(name):
    """Convert a subparam name to corresponding subparam class name."""
    # Replace _ with spaces
    name = name.replace("_", " ")
    # Capitalise
    name = name.title()
    # Remove spaces
    name = name.replace(" ", "")

    return "_" + name + "Parameters"

    # def electrolyte(self):
    """
    Parameters for the electrolyte
    *Chemistries*: lithium-ion, lead-acid
    """


#     return self._electrolyte
#
# @property
# def neg_electrode(self):

#     return self._neg_electrode
#
# @property
# def pos_electrode(self):
#
#     return self._pos_electrode
#
# @property
# def neg_reactions(self):

#     return self._neg_reactions
#
# @property
# def pos_reactions(self):

#     return self._pos_reactions
#
# @property
# def neg_volume_changes(self):

#     if self._chemistry != "lead-acid":
#         raise NotImplementedError
#
#     return self._neg_volume_changes
#
# @property
# def pos_volume_changes(self):
#     """
#     Parameters for volume changes in the positive electrode
#     *Chemistries*: lead-acid
#     """
#     if self._chemistry != "lead-acid":
#         raise NotImplementedError
#
#     return self._pos_volume_changes
#
# @property
# def temperature(self):

#     return self._temperature
#
# @property
# def lead_acid_misc(self):
#
#     if self._chemistry != "lead-acid":
#         raise NotImplementedError
#
#     return self._lead_acid_misc
class _GeometricParameters(object):
    """
     Geometric parameters.
     *Chemistries*: lithium-ion, lead-acid
     """

    def __init__(self, param):
        # Total width [m]
        self.L = param._raw["Ln"] + param._raw["Ls"] + param._raw["Lp"]
        # Area of the current collectors [m2]
        self.A_cc = param._raw["H"] * param._raw["W"]
        # Volume of a cell [m3]
        self.Vc = self.A_cc * self.L

        # Dimensionless half-width of negative electrode
        self.ln = param._raw["Ln"] / self.L
        # Dimensionless width of separator
        self.ls = param._raw["Ls"] / self.L
        # Dimensionless half-width of positive electrode
        self.lp = param._raw["Lp"] / self.L
        # Aspect ratio
        self.delta = self.L / param._raw["H"]


class _ElectricalParameters(object):
    """
    Parameters relating to the external electrical circuit.
    *Chemistries*: lithium-ion, lead-acid
    """

    def __init__(self, param):
        # Reference current density [A.m-2]
        self.ibar = abs(param.current["Ibar"]) / (
            param._raw["n_electrodes_parallel"] * param.geometric.A_cc
        )
        # C-rate [-]
        self.Crate = param.current["Ibar"] / param._raw["Q"]


class _ElectrolyteParameters(object):
    """
    Parameters for the electrolyte
    *Chemistries*: lithium-ion, lead-acid
    """

    def __init__(self, param):
        self.param = param
        # Effective reaction rates
        # Main reaction (neg) [-]
        self.sn = -(param._raw["spn"] + 2 * param._raw["tpw"]) / 2
        # Main reaction (pos) [-]
        self.sp = -(param._raw["spp"] + 2 * param._raw["tpw"]) / 2

        # Mesh-dependent parameters
        if param._mesh:
            self.s = np.concatenate(
                [
                    self.sn * np.ones_like(param._mesh.xn.centres),
                    np.zeros_like(param._mesh.xs.centres),
                    self.sp * np.ones_like(param._mesh.xp.centres),
                ]
            )
        else:
            self.s = "Mesh not set"

        # Diffusional C-rate: diffusion timescale/discharge timescale
        self.Cd = (
            (param.geometric.L ** 2)
            / param._func.D_hat(param._raw["cmax"])
            / (
                param._raw["cmax"]
                * param._raw["F"]
                * param.geometric.L
                / param.electrical.ibar
            )
        )

        # Initial conditions
        self.c0 = param._raw["q0"]

    # Dimensionless functions
    def D_eff(self, c, eps):
        return self.param._func.D_eff(self.param, c, eps)

    def kappa_eff(self, c, eps):
        return self.param._func.kappa_eff(self.param, c, eps)


class _NegElectrodeParameters(object):
    """
    Negative electrode physical parameters.
    *Chemistries*: lithium-ion, lead-acid
    """

    def __init__(self, param):
        # Effective lead conductivity (Bruggeman) [S.m-1]
        self.sigma_eff = param._raw["sigma_n"] * (1 - param._raw["epsnmax"]) ** 1.5

        # Dimensionless lead conductivity
        self.iota_s = (
            self.sigma_eff
            * param.scales["pot"]
            / (param.geometric.L * param.electrical.ibar)
        )
        # Dimensionless electrode capacity (neg)
        self.Qmax = param._raw["Qnmax_hat"] / (param._raw["cmax"] * param._raw["F"])


class _PosElectrodeParameters(object):
    """
    Positive electrode physical parameters.
    *Chemistries*: lithium-ion, lead-acid
    """

    def __init__(self, param):
        # Effective lead dioxide conductivity (Bruggeman) [S.m-1]
        self.sigma_eff = param._raw["sigma_p"] * (1 - param._raw["epspmax"]) ** 1.5

        # Dimensionless lead dioxide conductivity
        self.iota_s = (
            self.sigma_eff
            * param.scales["pot"]
            / (param.geometric.L * param.electrical.ibar)
        )
        # Dimensionless electrode capacity (pos)
        self.Qmax = param._raw["Qpmax_hat"] / (param._raw["cmax"] * param._raw["F"])


class _NegReactionsParameters(object):
    """
    Parameters for reactions in the negative electrode.
    *Chemistries*: lithium-ion, lead-acid
    """

    def __init__(self, param):
        self.param = param
        # Length
        self.l = param.geometric.ln
        # Dimensionless exchange-current density
        self.iota_ref = param._raw["jref_n"] / param.scales["jn"]
        # Dimensionless double-layer capacity
        self.gamma_dl = (
            param._raw["Cdl"]
            * param.scales["pot"]
            / param.scales["jn"]
            / (param.scales["time"] * 3600)
        )

    # Exchange-current density as function of concentration
    def j0(self, c):
        if self.param._chemistry == "lead-acid":
            return self.iota_ref * c

    # Dimensionless OCP
    def U(self, c):
        if self.param._chemistry == "lead-acid":
            return self.param._func.U_Pb(self.param, c)

    # Average interfacial current density
    def j_avg(self, t):
        return self.param.icell(t) / self.l


class _PosReactionsParameters(object):
    """
    Parameters for reactions in the positive electrode.
    *Chemistries*: lithium-ion, lead-acid
    """

    def __init__(self, param):
        self.param = param
        # Length
        self.l = param.geometric.lp
        # Dimensionless exchange-current density
        self.iota_ref = param._raw["jref_p"] / param.scales["jp"]
        # Dimensionless double-layer capacity
        self.gamma_dl = (
            param._raw["Cdl"]
            * param.scales["pot"]
            / param.scales["jp"]
            / (param.scales["time"] * 3600)
        )

    # Exchange-current density as function of concentration
    def j0(self, c):
        if self.param._chemistry == "lead-acid":
            return self.iota_ref * c ** 2 * self.param._func.cw(self.param, c)

    # Dimensionless OCP
    def U(self, c):
        if self.param._chemistry == "lead-acid":
            return self.param._func.U_PbO2(self.param, c)

    # Average interfacial current density
    def j_avg(self, t):
        return -self.param.icell(t) / self.l


class _NegVolumeChangesParameters(object):
    """
    Parameters for volume changes in the negative electrode.
    *Chemistries*: lead-acid
    """

    def __init__(self, param):
        # Net Molar Volume consumed in neg electrode [m3.mol-1]
        self.DeltaVsurf = param._raw["VPbSO4"] - param._raw["VPb"]

        # Dimensionless molar volume change (lead)
        self.beta_surf = param._raw["cmax"] * self.DeltaVsurf / 2


class _PosVolumeChangesParameters(object):
    """
    Parameters for volume changes in the positive electrode.
    *Chemistries*: lead-acid
    """

    def __init__(self, param):
        # Net Molar Volume consumed in pos electrode [m3.mol-1]
        self.DeltaVsurf = param._raw["VPbO2"] - param._raw["VPbSO4"]

        # Dimensionless molar volume change (lead dioxide)
        self.beta_surf = param._raw["cmax"] * self.DeltaVsurf / 2


class _TemperatureParameters(object):
    """
    Temperature parameters.
    *Chemistries*: lithium-ion, lead-acid
    """

    def __init__(self, param):
        # External temperature [K]
        self.T_inf = param._raw["T_ref"]


class _LeadAcidMiscParameters(object):
    """Miscellaneous parameters for lead-acid"""

    def __init__(self, param):
        self.param = param
        # Ratio of viscous pressure scale to osmotic pressure scale
        self.pi_os = (
            param._func.mu_hat(param._raw["cmax"])
            * param.scales["U_rxn"]
            * param.geometric.L
            / (
                param._raw["d"] ** 2
                * param._raw["R"]
                * param._raw["T_ref"]
                * param._raw["cmax"]
            )
        )
        # Dimensionless voltage cut-off
        self.voltage_cutoff = param.scales["pot"] * (
            param._raw["voltage_cutoff_circuit"] / param._raw["n_cells_series"]
            - (param._raw["U_PbO2_ref"] - param._raw["U_Pb_ref"])
        )

    # Dimensionless functions
    def chi(self, c):
        return self.param._func.chi(self.param, c)
