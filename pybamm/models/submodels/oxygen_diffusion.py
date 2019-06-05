#
# Equation classes for the oxygen concentration
#
import pybamm


class StefanMaxwell(pybamm.SubModel):
    """"Stefan-Maxwell Diffusion of oxygen in the oxygen.

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
        PDE system for Stefan-Maxwell diffusion in the oxygen

        Parameters
        ----------
        variables : dict
            Dictionary of symbols to use in the model
        reactions : dict
            Dictionary of reaction variables
        """
        # unpack variables
        c_ox = variables["Oxygen concentration"]
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
        N_ox_diff = -(epsilon ** param.b) * param.D_ox(c_ox) * pybamm.grad(c_ox)
        N_ox_conv = c_ox * v_box
        N_ox = N_ox_diff + N_ox_conv

        # Model
        source_terms = (
            sum(
                pybamm.Concatenation(
                    reaction["neg"]["s_ox"] * reaction["neg"]["aj"],
                    pybamm.Broadcast(0, ["separator"]),
                    reaction["pos"]["s_ox"] * reaction["pos"]["aj"],
                )
                for reaction in reactions.values()
            )
            / param.gamma_ox
        )
        self.rhs = {
            c_ox: (1 / epsilon)
            * (-pybamm.div(N_ox) / param.C_e + source_terms - c_ox * deps_dt)
        }

        self.initial_conditions = {c_ox: param.c_ox_init}
        self.boundary_conditions = {
            c_ox: {"left": (0, "Neumann"), "right": (0, "Neumann")}
        }
        self.variables = self.get_variables(c_ox, N_ox)

    def set_leading_order_system(self, variables, reactions):
        """
        ODE system for leading-order Stefan-Maxwell diffusion in the oxygen
        Parameters
        ----------
        variables : dict
            Dictionary of symbols to use in the model
        reactions : dict
            Dictionary of reaction variables
        """
        param = self.set_of_parameters
        c_ox = variables["Oxygen concentration"]
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
            param.l_n * rxn["neg"]["s_ox"] * rxn["neg"]["aj"]
            + param.l_p * rxn["pos"]["s_ox"] * rxn["pos"]["aj"]
            for rxn in reactions.values()
        )
        self.rhs = {
            c_ox: 1
            / (param.l_n * eps_n + param.l_s * eps_s + param.l_p * eps_p)
            * (source_terms - c_ox * (param.l_n * deps_n_dt + param.l_p * deps_p_dt))
        }
        self.initial_conditions = {c_ox: param.c_ox_init}

        # Variables
        whole_cell = ["negative electrode", "separator", "positive electrode"]
        N_ox = pybamm.Broadcast(0, whole_cell)
        c_ox_var = pybamm.Concatenation(
            pybamm.Broadcast(c_ox, ["negative electrode"]),
            pybamm.Broadcast(c_ox, ["separator"]),
            pybamm.Broadcast(c_ox, ["positive electrode"]),
        )
        self.variables = {
            **self.get_variables(c_ox_var, N_ox),
            "Average oxygen concentration": c_ox,
        }

    def get_variables(self, c_ox, N_ox):
        """
        Calculate dimensionless and dimensional variables for the oxygen diffusion
        submodel

        Parameters
        ----------
        c_ox : :class:`pybamm.Concatenation`
            Oxygen concentration
        N_ox : :class:`pybamm.Symbol`
            Flux of oxygen molecules

        Returns
        -------
        dict
            Dictionary {string: :class:`pybamm.Symbol`} of relevant variables
        """
        c_ox_typ = self.set_of_parameters.c_ox_typ

        if c_ox.domain == []:
            c_ox_n = pybamm.Broadcast(c_ox, domain=["negative electrode"])
            c_ox_s = pybamm.Broadcast(c_ox, domain=["separator"])
            c_ox_p = pybamm.Broadcast(c_ox, domain=["positive electrode"])
            c_ox = pybamm.Concatenation(c_ox_n, c_ox_s, c_ox_p)
        if N_ox.domain == []:
            N_ox_n = pybamm.Broadcast(N_ox, domain=["negative electrode"])
            N_ox_s = pybamm.Broadcast(N_ox, domain=["separator"])
            N_ox_p = pybamm.Broadcast(N_ox, domain=["positive electrode"])
            N_ox = pybamm.Concatenation(N_ox_n, N_ox_s, N_ox_p)

        c_ox_n, c_ox_s, c_ox_p = c_ox.orphans

        c_ox_av = pybamm.average(c_ox)

        return {
            "Oxygen concentration": c_ox,
            "Average oxygen concentration": c_ox_av,
            "Negative oxygen concentration": c_ox_n,
            "Separator oxygen concentration": c_ox_s,
            "Positive oxygen concentration": c_ox_p,
            "Reduced oxygen flux": N_ox,
            "Oxygen concentration [mol.m-3]": c_ox_typ * c_ox,
            "Average oxygen concentration [mol.m-3]": c_ox_typ * c_ox_av,
            "Negative oxygen concentration [mol.m-3]": c_ox_typ * c_ox_n,
            "Separator oxygen concentration [mol.m-3]": c_ox_typ * c_ox_s,
            "Positive oxygen concentration [mol.m-3]": c_ox_typ * c_ox_p,
        }
