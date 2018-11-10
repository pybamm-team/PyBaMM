import csv
import pandas as pd
import numpy as np
import warnings

def read_parameters_csv(filename):
    """Reads parameters from csv file into dict.

    Parameters
    ----------
    filename : string
        The name of the csv file to be opened.

    Returns
    -------
    dict
        {name: value} pairs for the parameters.

    """
    df = pd.read_csv(filename, comment='#', skip_blank_lines=True)
    # Drop rows that are all NaN (seems to not work with skip_blank_lines)
    df.dropna(how='all', inplace=True)
    return {k: v for (k,v) in zip(df.Name, df.Value)}

class Parameters:
    """
    The parameters for the simulation.
    """
    def __init__(self, current=None, optional_parameters={}):

        #######################################################################
        # Defaults ############################################################
        # Load default parameters from csv file
        default_parameters = read_parameters_csv(
            'input/parameters/default.csv')
        #######################################################################
        #######################################################################


        #######################################################################
        # Optional parameters #################################################
        # If optional_parameters is a filename, load from that filename
        if isinstance(optional_parameters, str):
            optional_parameters = read_parameters_csv(optional_parameters)
        else:
            # Otherwise, optional_parameters should be a dict
            assert isinstance(optional_parameters, dict), \
                """optional_parameters should be a filename (string) or a dict,
                but it is a '{}'""".format(type(optional_parameters))
        # Assign either default or optional values
        for param, default in default_parameters.items():
            try:
                self.__dict__[param] = optional_parameters[param]
            except KeyError:
                self.__dict__[param] = default
        #######################################################################
        #######################################################################


        #######################################################################
        # Input current #######################################################
        # Set default
        if current is None:
            current = {'Ibar': 1, 'type': 'constant'}
        self.current = current
        #######################################################################
        #######################################################################


        #######################################################################
        # Derived Parameters ##################################################
        # Geometric
        # Total width [m]
        self.L = self.Ln + self.Ls + self.Lp
        # Area of the current collectors [m2]
        self.A_cc = self.H * self.W
        # Volume of a cell [m3]
        self.Vc = self.A_cc * self.L

        # Electrical
        # Reference current density [A.m-2]
        self.ibar = abs(self.current['Ibar']
                        / (self.n_electrodes_parallel * self.A_cc))
        # C-rate [-]
        self.Crate = self.current['Ibar'] / self.Q

        # Effective reaction rates
        # Main reaction (neg) [-]
        self.sn = - (self.spn + 2*self.tpw) / 2
        # Main reaction (pos) [-]
        self.sp = - (self.spp + 2*self.tpw) / 2

        # Electrode physical properties
        # Effective lead conductivity (Bruggeman) [S.m-1]
        self.sigma_eff_n = self.sigma_n * (1-self.epsnmax)**1.5
        # Effective lead dioxide conductivity (Bruggeman) [S.m-1]
        self.sigma_eff_p = self.sigma_p * (1-self.epspmax)**1.5

        # Dimensional volume changes
        # Net Molar Volume consumed in neg electrode [m3.mol-1]
        self.DeltaVsurfN = self.VPbSO4 - self.VPb
        # Net Molar Volume consumed in pos electrode [m3.mol-1]
        self.DeltaVsurfP = self.VPbO2 - self.VPbSO4
        self.DeltaVliqN = self.Ve*(3-2*self.tpw) - 2*self.Vw
        self.DeltaVliqO2N = self.Ve*(2-2*self.tpw) - self.Vw
        self.DeltaVliqH2N = self.Ve*(2-2*self.tpw)
        self.DeltaVliqP = self.Ve*(1-2*self.tpw)
        self.DeltaVliqO2P = self.Ve*(2-2*self.tpw) - self.Vw
        self.DeltaVliqH2P = 0

        # Temperature
        # External temperature [K]
        self.T_inf = self.T_ref
        #######################################################################
        #######################################################################


        #######################################################################
        # Dimensionless Parameters ############################################
        self.scales = Scales(self)

        # Geometric
        # Dimensionless half-width of negative electrode
        self.ln = self.Ln / self.scales.length
        # Dimensionless width of separator
        self.ls = self.Ls / self.scales.length
        # Dimensionless half-width of positive electrode
        self.lp = self.Lp / self.scales.length
        # Aspect ratio
        self.delta = self.L / self.H
        # Width relative to height
        self.w = self.W / self.H

        # Exchange-current densities
        # Main reaction, negative electrode
        self.iota_ref_n = self.jref_n / self.scales.jn
        # Oxygen reaction, negative electrode
        self.iota_ref_O2_n = self.jrefO2_n / self.scales.jn
        # Hydrogen reaction, negative electrode
        self.iota_ref_H2_n = self.jrefH2_n / self.scales.jn
        # Main reaction, positive electrode
        self.iota_ref_p = self.jref_p / self.scales.jp
        # Oxygen reaction, positive electrode
        self.iota_ref_O2_p = self.jrefO2_p / self.scales.jp
        # Hydrogen reaction, positive electrode
        self.iota_ref_H2_p = self.jrefH2_p / self.scales.jp
        #######################################################################
        #######################################################################

    def set_mesh_dependent_parameters(self, mesh):
        """Create parameters that depend on the mesh
        (e.g. different in each electrode and separator).

        Parameters
        ----------
        mesh : pybamm.mesh.Mesh() instance
            The mesh on which to evaluate the parameters.

        """
        self.s = np.concatenate([self.sn*np.ones_like(mesh.xcn),
                                 np.zeros_like(mesh.xcs),
                                 self.sp*np.ones_like(mesh.xcp)])

    def xxx(self):

        # Cell heat capacity [kg.m2.s-2.K-1][J/K]
        self.Cp = 293
        # Density times specific heat capacity [kg.m-1.s-2.K-1][J/m^3K]
        self.rhocp = self.Cp/self.Vc
        # Thermal conductivity of lead [kg.m.s-3.K-1][W/mK]
        self.k = 34
        # Convective heat transfer coefficient [kg.s-3.K-1][W/m^2K]
        self.h = 2
        self.Tinit_hat = self.T_inf     # Initial temperature [K]

        # """Turn off side reactions?"""
        # self.jrefO2_n = self.jrefH2_n = self.jrefO2_p = self.jrefH2_p = 0
        # self.DeltaVliqN = self.DeltaVliqP = 0
        # self.satn0 = self.sats0 = self.satpw = 1

        """Dimensionless parameters"""
        # Diffusional C-rate: diffusion timescale/discharge timescale
        self.Cd = ((self.L**2)/self.D_hat(self.cmax)
                   / (self.cmax*self.F*self.L/self.ibar))

        # OCPs
        self.U_O2_n = self.F/(self.R*self.T_ref) * \
            (self.U_O2_hat - self.U_Pb_ref)    # Oxygen
        self.U_O2_p = self.F/(self.R*self.T_ref) * \
            (self.U_O2_hat - self.U_PbO2_ref)    # Oxygen
        self.U_H2_n = self.F/(self.R*self.T_ref) * \
            (self.U_H2_hat - self.U_Pb_ref)   # Hydrogen
        self.U_H2_p = self.F/(self.R*self.T_ref) * \
            (self.U_H2_hat - self.U_PbO2_ref)   # Hydrogen

        # Volume changes (minus sign comes from electron charge)
        self.beta_liq_n = self.cmax*self.DeltaVliqN/2       # Molar volume change
        self.beta_liq_O2_n = self.cmax*self.DeltaVliqO2N/2  # Molar volume change
        self.beta_liq_H2_n = self.cmax*self.DeltaVliqH2N/2  # Molar volume change
        self.beta_liq_p = self.cmax*self.DeltaVliqP/2       # Molar volume change
        self.beta_liq_O2_p = self.cmax*self.DeltaVliqO2P/2  # Molar volume change
        self.beta_liq_H2_p = self.cmax*self.DeltaVliqH2P/2  # Molar volume change
        self.beta_surf_n = self.cmax*self.DeltaVsurfN / \
            2    # Molar volume change (lead)
        self.beta_surf_p = self.cmax*self.DeltaVsurfP / \
            2    # Molar volume change (lead dioxide)

        # Electrode properties
        self.iota_s_n = (self.sigma_eff_n*self.R*self.T_ref
                         / (self.F*self.L)/self.ibar)    # Dimensionless lead conductivity
        self.iota_s_p = (self.sigma_eff_p*self.R*self.T_ref
                         / (self.F*self.L)/self.ibar)    # Dimensionless lead dioxide conductivity
        # Scaled electrode properties
        self.iota_s_n_bar = self.iota_s_n*self.delta**2*self.Cd
        self.iota_s_p_bar = self.iota_s_p*self.delta**2*self.Cd
        # Electrode capacity (neg)
        self.Qnmax = self.Qnmax_hat/(self.cmax*self.F)
        # Electrode capacity (pos)
        self.Qpmax = self.Qnmax_hat/(self.cmax*self.F)
        self.vert_cond_ratio = (self.ln*self.iota_s_n*self.delta**2 * self.lp*self.iota_s_p*self.delta**2
                                / (self.ln*self.iota_s_n*self.delta**2 + self.lp*self.iota_s_p*self.delta**2))  # Ratio of scaled conductivities

        # Other
        # Excluded volume fraction
        self.alpha = (2*self.Vw - self.Ve) * self.cmax
        # Dimensionless double-layer capacity (neg)
        self.gamma_dl_n = (self.Cdl*self.R*self.T_ref*self.Anmax*self.L
                           / (self.F*self.ibar)
                           / (self.cmax*self.F*self.L/self.ibar))
        # Dimensionless double-layer capacity (pos)
        self.gamma_dl_p = (self.Cdl*self.R*self.T_ref*self.Apmax*self.L
                           / (self.F*self.ibar)
                           / (self.cmax*self.F*self.L/self.ibar))
        # Dimensionless voltage cut-off
        self.voltage_cutoff = (self.F/(self.R*self.T_ref)
                               * (self.voltage_cutoff_circuit/6
                                  - (self.U_PbO2_ref - self.U_Pb_ref)))
        # Ratio of reference concentrations
        self.curlyC = 1/(self.cmax/self.cO2ref)
        # Ratio of reference diffusivities
        self.curlyD = self.D_hat(self.cmax)/self.DO2_hat(self.cO2ref)
        # Reaction velocity scale
        self.U_rxn = self.ibar/(self.cmax*self.F)
        # Ratio of viscous pressure scale to osmotic pressure scale
        self.pi_os = (self.mu_hat(self.cmax)*self.U_rxn*self.L
                      / (self.d**2*self.R*self.T_ref*self.cmax))

        # Temperature
        # Dimensionless reaction coefficient
        self.thetarxn = (self.R*self.T_ref*self.cmax
                         / (self.rhocp*(self.T_max-self.T_inf)))
        # Dimensionless thermal diffusion (should be small)
        self.thetadiff = (self.k*self.cmax*self.F
                          / (self.L*self.rhocp))
        # Dimensionless thermal convection
        self.thetaconv = (self.h*self.A_cc*self.cmax*self.F*self.L
                          / (self.Vc*self.rhocp*self.ibar))
        self.thetac = 0

        # Initial conditions
        # Dimensionless max capacity
        self.qmax = ((self.Ln * self.epsnmax
                      + self.Ls * self.epssmax
                      + self.Lp * self.epspmax
                      ) / self.L / (self.sp - self.sn))
        # if system == 'Bernardi':
        #     self.q0 = 0.39/4.94
        #     self.Un0 = 0.5
        #     self.Up0 = 0.5
        # else:
        self.Un0 = self.qmax/(self.Qnmax*self.ln)*(1-self.q0)
        self.Up0 = self.qmax/(self.Qpmax*self.lp)*(1-self.q0)
        self.epsDeltan = self.beta_surf_n/self.ln*self.qmax
        self.epsDeltap = self.beta_surf_p/self.lp*self.qmax
        # Initial pororsity (neg) [-]
        self.epsln0 = self.satn0*(self.epsnmax - self.epsDeltan*(1-self.q0))
        # Initial pororsity (sep) [-]
        self.epsls0 = self.sats0*self.epssmax
        # Initial pororsity (pos) [-]
        self.epslp0 = self.satpw*(self.epspmax - self.epsDeltap*(1-self.q0))
        self.epssolidn0 = 1 - (self.epsnmax - self.epsDeltan*(1-self.q0))
        self.epssolids0 = 1 - self.epssmax
        self.epssolidp0 = 1 - (self.epspmax - self.epsDeltap*(1-self.q0))
        self.c0 = self.q0
        logger.debug('Un0 = {}, Up0 = {}'.format(self.Un0, self.Up0))
        self.cO20 = 0
        self.T0 = (self.Tinit_hat - self.T_inf)/(self.T_max -
                                                 self.T_inf)    # Dimensionless initial temperature


    def Icircuit(self, t):
        """The current in the external circuit.

        Parameters
        ----------
        t : float or array_like, shape (n,)
            Time in *hours*.

        Returns
        -------
        float or array_like, shape (n,)
            The current at time(s) t, in Amps.

        """
        if self.current['type'] == 'constant':
            return 0*t + self.current['Ibar']

    def icell(self, t):
        """The dimensionless current function (could be some data)"""
        # This is a function of dimensionless time; Icircuit is a function of
        # time in *hours*
        return self.Icircuit(t*self.scales.time)/(8*self.A_cc)/self.ibar

    def D_hat(self, c):
        """
        Dimensional effective Fickian diffusivity in the electrolyte [m2.s-1].
        """
        return (1.75 + 260e-6*c)*1e-9

    def D_eff(self, c, eps):
        """Dimensionless effective Fickian diffusivity in the electrolyte."""
        return self.D_hat(c*self.cmax)/self.D_hat(self.cmax) * (eps**1.5)

    def DO2_hat(self, cO2):
        """
        Dimensional effective Fickian diffusivity of oxygen
        in the electrolyte [m2.s-1].
        """
        return 0*cO2 + 1e-9

    def DO2_eff(self, cO2, eps):
        """
        Dimensionless effective Fickian diffusivity of oxygen
        in the electrolyte.
        """
        return (self.DO2_hat(cO2*self.cO2ref) / self.DO2_hat(self.cO2ref)
                * (eps**1.5))

    def kappa_hat(self, c):
        """Dimensional effective conductivity in the electrolyte [S.m-1]"""
        return c * exp(6.23 - 1.34e-4*c - 1.61e-8 * c**2)*1e-4

    def kappa_eff(self, c, eps):
        """Dimensionless molar conductivity in the electrolyte"""
        kappa_scale = (self.F**2*self.cmax
                       * self.D_hat(self.cmax) / (self.R * self.T_ref))
        return self.kappa_hat(c*self.cmax)/kappa_scale * (eps**1.5)

    def chi_hat(self, c):
        """Dimensional Darken thermodynamic factor in the electrolyte [-]"""
        return 0.49+4.1e-4*c

    def chi(self, c):
        """Dimensionless Darken thermodynamic factor in the electrolyte"""
        chi_scale = 1/(2*(1-self.tpw))
        return self.chi_hat(c*self.cmax)/chi_scale / (1+self.alpha*c)

    def curlyK_hat(self, eps):
        """Dimensional permeability [m2]"""
        return eps**3 * self.d**2 / (180 * (1-eps)**2)

    def curlyK(self, eps):
        """Dimensionless permeability"""
        return self.curlyK_hat(eps) / self.d**2

    def mu_hat(self, c):
        """Dimensional viscosity of electrolyte [kg.m-1.s-1]"""
        return 0.89e-3 + 1.11e-7 * c + 3.29e-11 * c**2

    def mu(self, c):
        """Dimensionless viscosity of electrolyte"""
        return self.mu_hat(c*self.cmax)/self.mu_hat(self.cmax)

    def rho_hat(self, c):
        """Dimensional density of electrolyte [kg.m-3]"""
        return self.Mw/self.Vw*(1+(self.Me*self.Vw/self.Mw - self.Ve)*c)

    def rho(self, c):
        """Dimensionless density of electrolyte"""
        return self.rho_hat(c*self.cmax)/self.rho_hat(self.cmax)

    def cw_hat(self, c):
        """Dimensional solvent concentration [mol.m-3]"""
        return (1-c*self.Ve)/self.Vw

    def cw(self, c):
        """Dimensionless solvent concentration"""
        return self.cw_hat(c*self.cmax)/self.cw_hat(self.cmax)

    def dcwdc(self, c):
        """Dimensionless derivative of cw with respect to c"""
        return 0*c-self.Ve/self.Vw

    def m(self, c):
        """Dimensional electrolyte molar mass [mol.kg-1]"""
        return c*self.Vw/((1-c*self.Ve)*self.Mw)

    def dmdc(self, c):
        """Dimensional derivative of m with respect to c [kg-1]"""
        return self.Vw/((1-c*self.Ve)**2*self.Mw)

    def U_Pb(self, c):
        """Dimensionless OCP in the negative electrode"""
        m = self.m(c*self.cmax)  # dimensionless
        U = self.F/(self.R*self.T_ref)*(- 0.074 * np.log10(m)
                                        - 0.030 * np.log10(m)**2
                                        - 0.031 * np.log10(m)**3
                                        - 0.012 * np.log10(m)**4)
        return U

    def U_Pb_hat(self, c):
        """Dimensional OCP in the negative electrode [V]"""
        return (self.U_Pb_ref
                + self.R*self.T_ref/self.F * self.U_Pb(c/self.cmax))

    def dUPbdc(self, c):
        """Dimensionless derivative of U_Pb with respect to c"""
        m = self.m(c*self.cmax)  # dimensionless
        dUdm = self.F/(self.R*self.T_ref)*(
            - 0.074/m/np.log(10)
            - 0.030*2 * np.log(m)/(m*np.log(10)**2)
            - 0.031*3 * np.log(m)**2/m/np.log(10)**3
            - 0.012*4*np.log(m)**3/m/np.log(10)**4
            )
        dmdc = self.dmdc(c*self.cmax)*self.cmax  # dimensionless
        return dmdc*dUdm

    def U_PbO2(self, c):
        """Dimensionless OCP in the positive electrode"""
        m = self.m(c*self.cmax)
        U = self.F/(self.R*self.T_ref)*(0.074 * np.log10(m)
                                        + 0.033 * np.log10(m)**2
                                        + 0.043 * np.log10(m)**3
                                        + 0.022 * np.log10(m)**4)
        return U

    def U_PbO2_hat(self, c):
        """Dimensional OCP in the positive electrode [V]"""
        return (self.U_PbO2_ref
                + self.R*self.T_ref/self.F * self.U_PbO2(c/self.cmax))

    def dUPbO2dc(self, c):
        """Dimensionless derivative of U_PbO2 with respect to c"""
        m = self.m(c*self.cmax)  # dimensionless
        dUdm = self.F/(self.R*self.T_ref)*(
            0.074/m/np.log(10)
            + 0.033*2 * np.log(m)/(m*np.log(10)**2)
            + 0.043*3 * np.log(m)**2/m/np.log(10)**3
            + 0.022*4*np.log(m)**3/m/np.log(10)**4
            )
        dmdc = self.dmdc(c*self.cmax)*self.cmax  # dimensionless
        return dmdc*dUdm

class Scales:
    """Scales for non-dimensionalisation.

    Parameters
    ----------
    param : Parameters() instance
        The parameters from which to Calculate scales.

    """
    def __init__(self, param):
        # Length scale [m]
        self.length = param.L
        # Discharge time scale [h]
        self.time = param.cmax*param.F*param.L / param.ibar / 3600
        # Concentration scale [mol.m-3]
        self.conc = param.cmax
        # Current density scale [A.m-2]
        self.current = param.ibar
        # Interfacial current density scale (neg) [A.m-2]
        self.jn = param.ibar/(param.Anmax*param.L)
        # Interfacial current density scale (pos) [A.m-2]
        self.jp = param.ibar/(param.Apmax*param.L)
        # Interfacial area scale (neg) [m2.m-3]
        self.An = param.Anmax
        # Interfacial area scale (pos) [m2.m-3]
        self.Ap = param.Apmax
        # Interfacial area times current density [A.m-3]
        self.Aj = param.ibar/(param.L)
        # Voltage scale (thermal voltage) [V]
        self.pot = param.R*param.T_ref/param.F
        # Porosity, SOC scale [-]
        self.one = 1
        # Temperature scale [K]
        self.temp = param.T_max - param.T_inf

        # Combined scales
        self.It = self.current*self.time

        # Dictionary matching solution attributes
        # to re-dimensionalisation scales
        self.match = {'t': 'time',
                      'x': 'length',
                      'icell': 'current',
                      'Icircuit': 'current',
                      'intI': 'It',
                      'c0_v': 'conc',
                      'c1': 'conc',
                      'c': 'conc',
                      'c_avg': 'conc',
                      'cO2': 'concO2',
                      'cO2_avg': 'concO2',
                      'phi': ('pot', -param.U_Pb_ref),
                      'phis': ('pot', 0, param.U_PbO2_ref - param.U_Pb_ref),
                      'phisn': 'pot',
                      'phisp': ('pot', param.U_PbO2_ref - param.U_Pb_ref),
                      'xi': ('pot', -param.U_Pb_ref, param.U_PbO2_ref),
                      'xin': ('pot', -param.U_Pb_ref),
                      'xis': ('one', 0),
                      'xip': ('pot', param.U_PbO2_ref),
                      'V0': ('pot', param.U_PbO2_ref - param.U_Pb_ref),
                      'V1': ('pot', param.U_PbO2_ref - param.U_Pb_ref),
                      'V': ('pot', param.U_PbO2_ref - param.U_Pb_ref),
                      'V0circuit': ('pot',
                                    6*(param.U_PbO2_ref - param.U_Pb_ref)),
                      'Vcircuit': ('pot',
                                   6*(param.U_PbO2_ref - param.U_Pb_ref)),
                      'j': ('jn', 'jp'),
                      'jO2': ('jn', 'jp'),
                      'jH2': ('jn', 'jp'),
                      'jn': 'jn',
                      'js': 'one',
                      'jp': 'jp',
                      'jO2n': 'jn',
                      'jO2s': 'one',
                      'jO2p': 'jp',
                      'jH2n': 'jn',
                      'jH2s': 'one',
                      'jH2p': 'jp',
                      'A': ('An', 'Ap'),
                      'AO2': ('An', 'Ap'),
                      'AH2': ('An', 'Ap'),
                      'Adl': ('An', 'Ap'),
                      'An': 'An',
                      'As': 'one',
                      'Ap': 'Ap',
                      'AO2n': 'An',
                      'AO2s': 'one',
                      'AO2p': 'Ap',
                      'AH2n': 'An',
                      'AH2s': 'one',
                      'AH2p': 'Ap',
                      'Adln': 'An',
                      'Adls': 'one',
                      'Adlp': 'Ap',
                      'i': 'current',
                      'i_n': 'current',
                      'i_p': 'current',
                      'isolid': 'current',
                      'q': 'one',
                      'U': 'one',
                      'Un': 'one',
                      'Us': 'one',
                      'Up': 'one',
                      'T': ('temp', param.T_inf),
                      }
