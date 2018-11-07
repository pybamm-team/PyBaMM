import numpy as np

# csv file
TUNABLE_PARAMETERS_DEFAULTS = {'q0': 1,
                               }

class Parameters:
    """
    The parameters for the simulation.
    """
    def __init__(self):
        self.ln = 1/3
        self.ls = 1/3
        self.lp = 1/3

    def xxx(self, tunable_parameters):

        for param, default in TUNABLE_PARAMETERS_DEFAULTS.items():
            try:
                self.__dict__[param] = tunable_parameters[param]
            except KeyError:
                self.__dict__[param] = default

        # Check we aren't trying to tune un-tuneable parameters
        for param_name in tunable_parameters.keys():
            if param_name not in TUNABLE_PARAMETERS_DEFAULTS.keys():
                warnings.warn("{!s} is not a tuneable parameter, ignoring"
                              .format(param_name))

        """Defaults and tolerances."""
        if fit is None:
            fit = {'q0': 1,
                   'epsnmax': 0.53,
                   'epssmax': 0.92,
                   'epspmax': 0.57,
                   'cmax': 5.65,
                   'jref_n': .06,
                   'jref_p': .004}
        if simplifications is None:
            simplifications = {'dilute': False,
                               'no_side': False}
        self.system = system
        for k, v in simplifications.items():
            setattr(self, k, v)

        # Tolerances
        self.c_tol = 0.01

        """Dimensional parameters"""
        # Constants
        self.R = 8.314                  # Gas constant [J.K-1.mol-1]
        self.F = 96487                  # Faraday constant [C.mol-1]
        self.T_ref = 21.7 + 273.15          # Reference temperature [K]

        # Lengths
        if system == 'BBOXX':
            # Half-width of negative electrode [m]
            self.Ln = (1.8e-3)/2
            self.Ls = 1.5e-3                # Width of separator [m]
            # Half-width of positive electrode [m]
            self.Lp = (2.5e-3)/2
            self.H = 0.114                  # Cell height [m]
            self.W = .065                   # Cell depth [m]
        elif system == 'Srinivasan':
            # Half-width of negative electrode [m]
            self.Ln = (0.785e-3)
            self.Ls = 1.146e-3                # Width of separator [m]
            # Half-width of positive electrode [m]
            self.Lp = (1.145e-3)
            self.H = 0.127                  # Cell height [m]
            self.W = 0.1016                   # Cell depth [m]
        elif system == 'Bernardi':
            self.Ln = 1.4e-3            # Half-width of negative electrode [m]
            self.Ls = 1.3e-3                # Width of separator [m]
            self.Lp = 0.9e-3            # Half-width of positive electrode [m]
            self.H = 0.127                  # Cell height [m] (guess)
            self.W = 109.8e-4/self.H                   # Cell depth [m]
        self.L = self.Ln+self.Ls+self.Lp    # Total width [m]
        # Dimensionless
        self.ln = self.Ln/self.L        # Dimensionless half-width of negative electrode
        self.ls = self.Ls/self.L        # Dimensionless width of separator
        self.lp = self.Lp/self.L        # Dimensionless half-width of positive electrode

        # If we only needed the grid parameters, return
        if gridpars_only:
            return None

        # Current collectors
        self.A_cs = self.H*self.W       # Area of the current collectors [m2]
        self.Vc = self.A_cs*self.L      # Volume of a cell [m3]
        # Specified scale for the current [A.m-2]
        self.ibar = abs(Ibar/(8*self.A_cs))
        self.Icircuit = Icircuit
        self.voltage_cutoff_circuit = 10  # Voltage cut-off for the circuit[V]
        # Manufacturer-specified capacity [Ah]
        self.Q = 17
        self.Crate = Ibar/self.Q        # C-rate [-]

        # Microstructure
        # Negative electrode surface area density [m-1] (or 1e4 or 1e6?)
        self.Anmax = 2.3e6
        # Positive electrode surface area density [m-1]
        self.Apmax = 23e6
        # Max porosity of negative electrode [-]
        self.epsnmax = fit['epsnmax']
        self.epssmax = fit['epssmax']       # Max porosity of separator [-]
        # Max porosity of positive electrode [-]
        self.epspmax = fit['epspmax']
        self.xi = 0.6                       # Morphology factor
        self.Qnmax_hat = 3473e6           # Electrode capacity (neg) [C.m-3]
        self.Qpmax_hat = 2745e6           # Electrode capacity (pos) [C.m-3]

        # Stoichiometric coefficients
        self.spn = -1                   # s_+ in the negative electrode [-]
        self.spp = -3                   # s_+ in the positive electrode [-]

        # Electrolyte physical properties
        # Maximum electrolye concentration [mol.m-3]
        self.cmax = fit['cmax']*1e3
        self.tp0 = 0.7                  # Cation transference number [-]
        # Partial molar volume of water [m3.mol-1]
        self.Vw = 17.5e-6
        # Partial molar volume of electrolyte [m3.mol-1]
        self.Ve = 45e-6
        self.Mw = 18.01e-3              # Molar mass of water [kg.mol-1]

        # Electrode physical properties
        self.VPb = 207e-3/11.34e3       # Molar volume of lead [m3.mol-1]
        # Molar volume of lead dioxide [m3.mol-1]
        self.VPbO2 = 239e-3/9.38e3
        # Molar volume of lead sulfate [m3.mol-1]
        self.VPbSO4 = 303e-3/6.29e3
        # Effective lead conductivity [S/m-1]
        self.sigma_eff_n = 4.8e6*(1-self.epsnmax)**1.5
        # Effective lead dioxide conductivity [S/m-1]
        self.sigma_eff_p = 8e4*(1-self.epspmax)**1.5
        self.d = 1e-7                   # Pore size [m]

        # Gas phyisical properties
        self.cO2ref = 1e3               # Reference O2 concentration [mol.m-3]

        # Dimensional volume changes
        # Net Molar Volume consumed in neg electrode [m3.mol-1]
        self.DeltaVsurfN = self.VPbSO4 - self.VPb
        # Net Molar Volume consumed in pos electrode [m3.mol-1]
        self.DeltaVsurfP = self.VPbO2 - self.VPbSO4
        self.DeltaVliqN = self.Ve*(3-2*self.tp0) - 2*self.Vw
        self.DeltaVliqO2N = self.Ve*(2-2*self.tp0) - self.Vw
        self.DeltaVliqH2N = self.Ve*(2-2*self.tp0)
        self.DeltaVliqP = self.Ve*(1-2*self.tp0)
        self.DeltaVliqO2P = self.Ve*(2-2*self.tp0) - self.Vw
        self.DeltaVliqH2P = 0

        # Initial saturations
        self.satn0 = 0.85
        self.sats0 = 0.93
        self.satp0 = 0.85

        # Butler-Volmer
        # Reference exchange-current densities [A.m-2]
        if system == 'Bernardi':
            self.jref_n = 1e5/self.Anmax
            self.jref_p = 1.1e5/self.Apmax
            self.jrefO2_n = 1.15e-25/self.Anmax         # O2 neg
            self.jrefH2_n = 0  # 1.56e-11        # H2 neg
            self.jrefO2_p = 1.15e-13/self.Apmax         # O2 pos
            self.jrefH2_p = 0               # H2 pos
        else:
            self.jref_n = fit['jref_n']     # main neg
            self.jref_p = fit['jref_p']     # main pos
            self.jrefO2_n = 2.5e-32         # O2 neg
            self.jrefH2_n = 0  # 1.56e-11        # H2 neg
            self.jrefO2_p = 2.5e-23         # O2 pos
            self.jrefH2_p = 0               # H2 pos
        # Open-circuit potentials [V vs SHE]
        self.U_Pb_ref = -0.294          # Lead
        self.U_PbO2_ref = 1.628         # Lead dioxide
        self.U_O2_hat = 1.229           # Oxygen
        self.U_H2_hat = 0               # Hydrogen

        # Double-layer capacity [F.m-2]
        self.Cdl = .2

        # Temperature
        self.T_inf = self.T_ref         # External temperature [K]
        self.T_max = 273.15+60          # Maximum temperature [K]
        # Cell heat capacity [kg.m2.s-2.K-1][J/K]
        self.Cp = 293
        # Density times specific heat capacity [kg.m-1.s-2.K-1][J/m^3K]
        self.rhocp = self.Cp/self.Vc
        # Thermal conductivity of lead [kg.m.s-3.K-1][W/mK]
        self.k = 34
        # Convective heat transfer coefficient [kg.s-3.K-1][W/m^2K]
        self.h = 2
        self.Tinit_hat = self.T_inf     # Initial temperature [K]
        # Scales
        self.scales = Scales(self)

        # """Turn off side reactions?"""
        # self.jrefO2_n = self.jrefH2_n = self.jrefO2_p = self.jrefH2_p = 0
        # self.DeltaVliqN = self.DeltaVliqP = 0
        # self.satn0 = self.sats0 = self.satp0 = 1

        """Dimensionless parameters"""
        # Geometry
        self.delta = self.L/self.H
        self.w = self.W/self.H

        # Diffusional C-rate: diffusion timescale/discharge timescale
        self.Cd = ((self.L**2)/self.D_hat(self.cmax)
                   / (self.cmax*self.F*self.L/self.ibar))

        # Reactions
        # Dimensionless rection rate (neg)
        self.sn = -(self.spn + 2*self.tp0)/2
        # Dimensionless rection rate (pos)
        self.sp = -(self.spp + 2*self.tp0)/2
        # Dimensionless reaction rate (O2 neg)
        self.sO2n = 1-self.tp0
        # Dimensionless reaction rate (O2 pos)
        self.sO2p = 1-self.tp0
        # Dimensionless reaction rate (H2 neg)
        self.sH2n = 1-self.tp0
        # Dimensionless reaction rate (H2 pos)
        self.sH2p = 1-self.tp0
        # Dimensionless reaction rate (O2 reaction)
        self.sO2siden = 1/4
        # Dimensionless reaction rate (O2 reaction)
        self.sO2sidep = 1/4

        # Exchange-current densities
        # Dimensionless exchange-current density (neg)
        self.iota_ref_n = self.jref_n/(self.ibar/(self.Anmax*self.L))
        # Dimensionless exchange-current density (neg)
        self.iota_ref_O2_n = self.jrefO2_n/(self.ibar/(self.Anmax*self.L))
        # Dimensionless exchange-current density (neg)
        self.iota_ref_H2_n = self.jrefH2_n/(self.ibar/(self.Anmax*self.L))
        # Dimensionless exchange-current density (pos)
        self.iota_ref_p = self.jref_p/(self.ibar/(self.Apmax*self.L))
        # Dimensionless exchange-current density (pos)
        self.iota_ref_O2_p = self.jrefO2_p/(self.ibar/(self.Apmax*self.L))
        # Dimensionless exchange-current density (pos)
        self.iota_ref_H2_p = self.jrefH2_p/(self.ibar/(self.Apmax*self.L))

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
        self.alpha = (2*self.Vw - self.Ve) * \
            self.cmax    # Excluded volume fraction
        self.gamma_dl_n = (self.Cdl*self.R*self.T_ref*self.Anmax*self.L
                           / (self.F*self.ibar)
                           / (self.cmax*self.F*self.L/self.ibar))   # Dimensionless double-layer capacity (neg)
        self.gamma_dl_p = (self.Cdl*self.R*self.T_ref*self.Apmax*self.L
                           / (self.F*self.ibar)
                           / (self.cmax*self.F*self.L/self.ibar))   # Dimensionless double-layer capacity (pos)
        self.voltage_cutoff = (self.F/(self.R*self.T_ref)
                               * (self.voltage_cutoff_circuit/6
                                  - (self.U_PbO2_ref - self.U_Pb_ref)))  # Dimensionless voltage cut-off
        # Ratio of reference concentrations
        self.curlyC = 1/(self.cmax/self.cO2ref)
        # Ratio of reference diffusivities
        self.curlyD = self.D_hat(self.cmax)/self.DO2_hat(self.cO2ref)
        # Reaction velocity scale
        self.U_rxn = self.ibar/(self.cmax*self.F)
        self.pi_os = (self.mu_hat(self.cmax)*self.U_rxn*self.L
                      / (self.d**2*self.R*self.T_ref*self.cmax))         # Ratio of viscous pressure scale to osmotic pressure scale

        # Temperature
        self.thetarxn = (self.R*self.T_ref*self.cmax
                         / (self.rhocp*(self.T_max-self.T_inf)))     # Dimensionless reaction coefficient
        self.thetadiff = (self.k*self.cmax*self.F
                          / (self.L*self.rhocp))     # Dimensionless thermal diffusion (should be small)
        self.thetaconv = (self.h*self.A_cs*self.cmax*self.F*self.L
                          / (self.Vc*self.rhocp*self.ibar))  # Dimensionless thermal convection
        self.thetac = 0

        # Initial conditions
        self.q0 = fit['q0']       # Initial SOC [-]
        self.set_initial_conditions()
        # for k in [self.Cd, self.iota_s_n, self.iota_s_p, self.iota_ref_n, self.iota_ref_p,
        #           self.beta_surf_n, self.beta_surf_p, self.gamma_dl_n, self.gamma_dl_p]:
        #     print(k)
        # pdb.set_trace()

    def icell(self, t):
        """The dimensionless current function (could be some data)"""
        # This is a function of dimensionless time; Icircuit is a function of
        # time in *hours*
        return self.Icircuit(t*self.scales.time)/(8*self.A_cs)/self.ibar

    def set_initial_conditions(self):
        self.qmax = ((self.Ln*self.epsnmax+self.Ls*self.epssmax+self.Lp*self.epspmax)/self.L
                     / (self.sp-self.sn))   # Dimensionless max capacity
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
        self.epslp0 = self.satp0*(self.epspmax - self.epsDeltap*(1-self.q0))
        self.epssolidn0 = 1 - (self.epsnmax - self.epsDeltan*(1-self.q0))
        self.epssolids0 = 1 - self.epssmax
        self.epssolidp0 = 1 - (self.epspmax - self.epsDeltap*(1-self.q0))
        self.c0 = self.q0
        logger.debug('Un0 = {}, Up0 = {}'.format(self.Un0, self.Up0))
        self.cO20 = 0
        self.T0 = (self.Tinit_hat - self.T_inf)/(self.T_max -
                                                 self.T_inf)    # Dimensionless initial temperature

    def D_hat(self, c):
        """Dimensional effective Fickian diffusivity in the electrolyte [m2.s-1]"""
        return (1.75 + 260e-6*c)*1e-9

    def D_eff(self, c, eps):
        """Dimensionless effective Fickian diffusivity in the electrolyte."""
        return self.D_hat(c*self.cmax)/self.D_hat(self.cmax) * (eps**1.5)

    def dDhatdc(self, c):
        return 260e-6*1e-9

    def dDeffdc(self, c, eps):
        return self.cmax*self.dDhatdc(c*self.cmax)/self.D_hat(self.cmax) * (eps**1.5)

    def dDeffdeps(self, c, eps):
        return self.D_hat(c*self.cmax)/self.D_hat(self.cmax) * (1.5*eps**0.5)

    def DO2_hat(self, cO2):
        # Must have the same shape as c
        return 0*cO2 + 1e-9

    def DO2_eff(self, cO2, eps):
        return self.DO2_hat(cO2*self.cO2ref)/self.DO2_hat(self.cO2ref) * (eps**1.5)

    def dDO2hatdc(self, c):
        return 0

    def dDO2effdc(self, c, eps):
        return self.cmax*self.dDO2hatdc(c*self.cmax)/self.DO2_hat(self.cmax) * (eps**1.5)

    def dDO2effdeps(self, c, eps):
        return self.DO2_hat(c*self.cmax)/self.DO2_hat(self.cmax) * (1.5*eps**0.5)

    def kappa_hat(self, c, symbolic=False):
        """Dimensional effective conductivity in the electrolyte [S.m-1]"""
        if symbolic:
            exp = sp.exp
        else:
            exp = np.exp
        return c * exp(6.23 - 1.34e-4*c - 1.61e-8 * c**2)*1e-4

    def kappa_eff(self, c, eps, symbolic=False):
        """Dimensionless molar conductivity in the electrolyte"""
        kappa_scale = self.F**2*self.cmax * \
            self.D_hat(self.cmax)/(self.R*self.T_ref)
        return self.kappa_hat(c*self.cmax, symbolic)/kappa_scale * (eps**1.5)

    def dkappahatdc(self, c, symbolic=False):
        if symbolic:
            exp = sp.exp
        else:
            exp = np.exp
        return (exp(6.23 - 1.34e-4*c - 1.61e-8 * c**2) * (1 + c*(-1.34e-4 - 2*1.61e-8*c)))*1e-4

    def dkappaeffdc(self, c, eps):
        kappa_scale = self.F**2*self.cmax * \
            self.D_hat(self.cmax)/(self.R*self.T_ref)
        return self.cmax*self.dkappahatdc(c*self.cmax)/kappa_scale * (eps**1.5)

    def dkappaeffdeps(self, c, eps):
        kappa_scale = self.F**2*self.cmax * \
            self.D_hat(self.cmax)/(self.R*self.T_ref)
        return self.kappa_hat(c*self.cmax)/kappa_scale * (1.5*eps**0.5)

    def chi_hat(self, c):
        """Dimensional Darken thermodynamic factor in the electrolyte [-]"""
        return 0.49+4.1e-4*c

    def dchihatdc(self, c):
        return 4.1e-4

    def chi(self, c):
        """Dimensionless Darken thermodynamic factor in the electrolyte"""
        chi_scale = 1/(2*(1-self.tp0))
        return self.chi_hat(c*self.cmax)/chi_scale / (1+self.alpha*c)

    def dchidc(self, c):
        chi_scale = 1/(2*(1-self.tp0))
        deriv = (self.cmax*self.dchihatdc(c*self.cmax) / (1+self.alpha*c)
                 - self.alpha*self.chi_hat(c*self.cmax) / (1+self.alpha*c)**2)/chi_scale
        return deriv

    def curlyK_hat(self, eps):
        """Dimensional permeability [m2]"""
        return eps**3 * self.d**2 / (180 * (1-eps)**2)

    def curlyK(self, eps):
        """Dimensionless permeability"""
        return self.curlyK_hat(eps)/self.d**2

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
        # Must have the same shape as c
        return 0*c-self.Ve/self.Vw

    def m(self, c):
        """Dimensional electrolyte molar mass [mol.kg-1]"""
        return c*self.Vw/((1-c*self.Ve)*self.Mw)

    def dmdc(self, c):
        """Dimensional derivative of m with respect to c [kg-1]"""
        return self.Vw/((1-c*self.Ve)**2*self.Mw)

    def U_Pb(self, c, symbolic=False):
        """Dimensionless OCP in the negative electrode"""
        if symbolic:
            def log10(x): return sp.log(x, 10)
        else:
            log10 = np.log10
        m = self.m(c*self.cmax)  # dimensionless
        U = self.F/(self.R*self.T_ref)*(- 0.074*log10(m)
                                        - 0.030*log10(m)**2
                                        - 0.031*log10(m)**3
                                        - 0.012*log10(m)**4)
        return U

    def U_Pb_hat(self, c):
        """Dimensional OCP in the negative electrode [V]"""
        return self.U_Pb_ref + self.R*self.T_ref/self.F * self.U_Pb(c/self.cmax)

    def dUPbdc(self, c):
        """Dimensionless derivative of U_Pb with respect to c"""
        m = self.m(c*self.cmax)  # dimensionless
        dUdm = self.F/(self.R*self.T_ref)*(- 0.074/m/np.log(10)
                                           - 0.030*2 *
                                           np.log(m)/(m*np.log(10)**2)
                                           - 0.031*3 *
                                           np.log(m)**2/m/np.log(10)**3
                                           - 0.012*4*np.log(m)**3/m/np.log(10)**4)
        dmdc = self.dmdc(c*self.cmax)*self.cmax  # dimensionless
        return dmdc*dUdm

    def U_PbO2(self, c, symbolic=False):
        """Dimensionless OCP in the positive electrode"""
        if symbolic:
            def log10(x): return sp.log(x, 10)
        else:
            log10 = np.log10
        m = self.m(c*self.cmax)
        U = self.F/(self.R*self.T_ref)*(0.074*log10(m)
                                        + 0.033*log10(m)**2
                                        + 0.043*log10(m)**3
                                        + 0.022*log10(m)**4)
        return U

    def U_PbO2_hat(self, c):
        """Dimensional OCP in the positive electrode [V]"""
        return self.U_PbO2_ref + self.R*self.T_ref/self.F * self.U_PbO2(c/self.cmax)

    def dUPbO2dc(self, c):
        """Dimensionless derivative of U_Pb with respect to c"""
        m = self.m(c*self.cmax)  # dimensionless
        dUdm = self.F/(self.R*self.T_ref)*(0.074/m/np.log(10)
                                           + 0.033*2 *
                                           np.log(m)/(m*np.log(10)**2)
                                           + 0.043*3 *
                                           np.log(m)**2/m/np.log(10)**3
                                           + 0.022*4*np.log(m)**3/m/np.log(10)**4)
        dmdc = self.dmdc(c*self.cmax)*self.cmax  # dimensionless
        return dmdc*dUdm
