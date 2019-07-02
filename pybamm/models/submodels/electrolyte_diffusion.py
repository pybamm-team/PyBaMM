#
# Equation classes for the electrolyte concentration
#
import pybamm


class StefanMaxwell(pybamm.SubModel):
    """"A class that generates the expression tree for Stefan-Maxwell Diffusion in the
    electrolyte.

    Parameters
    ----------
    set_of_parameters : parameter class
        The parameters to use for this submodel

    *Extends:* :class:`pybamm.SubModel`
    """

    def __init__(self, set_of_parameters):
        super().__init__(set_of_parameters)

    def set_differential_system(self, variables, reactions):
        """
        PDE system for Stefan-Maxwell diffusion in the electrolyte

        Parameters
        ----------
        variables : dict
            Dictionary of symbols to use in the model
        reactions : dict
            Dictionary of reaction variables
        """
        # unpack variables
        c_e = variables["Electrolyte concentration"]
        param = self.set_of_parameters

        # if porosity is not provided, use the input parameter
        try:
            epsilon = variables["Porosity"]
            deps_dt = sum(rxn["porosity change"] for rxn in reactions.values())
        except KeyError:
            epsilon = param.epsilon
            deps_dt = pybamm.Scalar(0)

        # Use convection velocity if it exists, otherwise set to zero
        v_box = variables.get("Volume-averaged velocity", pybamm.Scalar(0))

        # Flux
        N_e_diff = -(epsilon ** param.b) * param.D_e(c_e) * pybamm.grad(c_e)
        N_e_conv = c_e * v_box
        N_e = N_e_diff + N_e_conv

        # Model
        source_terms = (
            sum(
                pybamm.Concatenation(
                    reaction["neg"]["s"] * reaction["neg"]["aj"],
                    pybamm.Broadcast(0, ["separator"]),
                    reaction["pos"]["s"] * reaction["pos"]["aj"],
                )
                for reaction in reactions.values()
            )
            / param.gamma_e
        )
        self.rhs = {
            c_e: (1 / epsilon)
            * (-pybamm.div(N_e) / param.C_e + source_terms - c_e * deps_dt)
        }

        self.initial_conditions = {c_e: param.c_e_init}
        self.boundary_conditions = {
            c_e: {"left": (0, "Neumann"), "right": (0, "Neumann")}
        }
        self.variables = self.get_variables(c_e, N_e)

        # Cut off if concentration goes too small
        # (open-circuit potential poorly defined)
        self.events["Zero electrolyte concentration cut-off"] = pybamm.min(c_e) - 0.002

    def set_leading_order_system(self, variables, reactions):
        """
        ODE system for leading-order Stefan-Maxwell diffusion in the electrolyte
        Parameters
        ----------
        variables : dict
            Dictionary of symbols to use in the model
        reactions : dict
            Dictionary of reaction variables
        """
        param = self.set_of_parameters
        c_e = variables["Electrolyte concentration"]
        try:
            epsilon = variables["Porosity"]
            deps_n_dt = sum(rxn["neg"]["deps_dt"] for rxn in reactions.values())
            deps_p_dt = sum(rxn["pos"]["deps_dt"] for rxn in reactions.values())
        except KeyError:
            epsilon = param.epsilon
            deps_n_dt = pybamm.Scalar(0)
            deps_p_dt = pybamm.Scalar(0)

        eps_n, eps_s, eps_p = [e.orphans[0] for e in epsilon.orphans]

        # Model
        source_terms = sum(
            param.l_n * rxn["neg"]["s"] * rxn["neg"]["aj"]
            + param.l_p * rxn["pos"]["s"] * rxn["pos"]["aj"]
            for rxn in reactions.values()
        )
        self.rhs = {
            c_e: 1
            / (param.l_n * eps_n + param.l_s * eps_s + param.l_p * eps_p)
            * (source_terms - c_e * (param.l_n * deps_n_dt + param.l_p * deps_p_dt))
        }
        self.initial_conditions = {c_e: param.c_e_init}

        # Variables
        whole_cell = ["negative electrode", "separator", "positive electrode"]
        N_e = pybamm.Broadcast(0, whole_cell)
        c_e_var = pybamm.Concatenation(
            pybamm.Broadcast(c_e, ["negative electrode"]),
            pybamm.Broadcast(c_e, ["separator"]),
            pybamm.Broadcast(c_e, ["positive electrode"]),
        )
        self.variables = {
            **self.get_variables(c_e_var, N_e),
            "Average electrolyte concentration": c_e,
        }

        # Cut off if concentration goes too small
        # (open-circuit potential poorly defined)
        self.events["Zero electrolyte concentration cut-off"] = pybamm.min(c_e) - 0.002

    def get_variables(self, c, N, species="electrolyte"):
        """
        Calculate dimensionless and dimensional variables for the electrolyte diffusion
        submodel

        Parameters
        ----------
        c : :class:`pybamm.Concatenation`
            Concentration of ions/molecules
        N : :class:`pybamm.Symbol`
            Flux of ioins/molecules
        species : str, optional
            The name of the species to set variables for (default is "electrolyte")

        Returns
        -------
        dict
            Dictionary {string: :class:`pybamm.Symbol`} of relevant variables
        """
        if species == "electrolyte":
            c_typ = self.set_of_parameters.c_e_typ
            flux_species = "cation"
        elif species == "oxygen":
            c_typ = self.set_of_parameters.c_ox_typ
            flux_species = "oxygen"

        if c.domain == []:
            c_n = pybamm.Broadcast(c, domain=["negative electrode"])
            c_s = pybamm.Broadcast(c, domain=["separator"])
            c_p = pybamm.Broadcast(c, domain=["positive electrode"])
            c = pybamm.Concatenation(c_n, c_s, c_p)
        if N.domain == []:
            N_n = pybamm.Broadcast(N, domain=["negative electrode"])
            N_s = pybamm.Broadcast(N, domain=["separator"])
            N_p = pybamm.Broadcast(N, domain=["positive electrode"])
            N = pybamm.Concatenation(N_n, N_s, N_p)

        c_n, c_s, c_p = c.orphans

        c_av = pybamm.average(c)

        return {
            species.capitalize() + " concentration": c,
            "Average " + species + " concentration": c_av,
            "Negative " + species + " concentration": c_n,
            "Separator " + species + " concentration": c_s,
            "Positive " + species + " concentration": c_p,
            "Reduced " + flux_species + " flux": N,
            species.capitalize() + " concentration [mol.m-3]": c_typ * c,
            "Average " + species + " concentration [mol.m-3]": c_typ * c_av,
            "Negative " + species + " concentration [mol.m-3]": c_typ * c_n,
            "Separator " + species + " concentration [mol.m-3]": c_typ * c_s,
            "Positive " + species + " concentration [mol.m-3]": c_typ * c_p,
        }
