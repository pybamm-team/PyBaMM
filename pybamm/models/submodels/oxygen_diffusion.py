#
# Equation classes for the oxygen concentration
#
import pybamm


class StefanMaxwell(pybamm.electrolyte_diffusion.StefanMaxwell):
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
        import ipdb

        ipdb.set_trace()
        source_terms = sum(
            param.l_n * rxn["neg"]["s_ox"] * rxn["neg"]["aj"].orphans[0]
            + param.l_p * rxn["pos"]["s_ox"] * rxn["pos"]["aj"].orphans[0]
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
        """ See :meth:`pybamm.electrolyte_diffusion.StefanMaxwell.get_variables`. """
        return super().get_variables(c_ox, N_ox, species="oxygen")
