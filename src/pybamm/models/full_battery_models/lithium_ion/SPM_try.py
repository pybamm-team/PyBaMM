import pybamm
import numpy as np

class singleparticletry(pybamm.lithium_ion.base_lithium_ion_model):
    def __init__(self,options,name="SEItry"):

        options = options or {}
        
        super().__init__(options, name)
        
        param = self.param
        R = pybamm.Parameter("Particle radius [m]")
        D = pybamm.Parameter("Diffusion coefficient [m2.s-1]")
        j = pybamm.Parameter("Interfacial current density [A.m-2]")
        F = pybamm.Parameter("Faraday constant [C.mol-1]")
        c0 = pybamm.Parameter("Initial concentration [mol.m-3]")
        c = pybamm.Variable("Concentration [mol.m-3]", domain="negative particle")

        # governing equations
        N = -D * pybamm.grad(c)  # flux
        dcdt = -pybamm.div(N)
        self.rhs = {c: dcdt}

        # boundary conditions
        lbc = pybamm.Scalar(0)
        rbc = -j / F / D
        self.boundary_conditions = {c: {"left": (lbc, "Neumann"), "right": (rbc, "Neumann")}}

        # initial conditions
        self.initial_conditions = {c: c0}

        self.variables = {
            "Concentration [mol.m-3]": c,
            "Surface concentration [mol.m-3]": pybamm.surf(c),
            "Flux [mol.m-2.s-1]": N,
        }