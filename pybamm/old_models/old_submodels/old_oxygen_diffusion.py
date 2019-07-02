#
# Equation classes for the oxygen concentration
#
import pybamm


class OldStefanMaxwell(pybamm.old_electrolyte_diffusion.OldStefanMaxwell):
    """"Stefan-Maxwell Diffusion of oxygen in the oxygen.

    Parameters
    ----------
    set_of_parameters : parameter class
        The parameters to use for this submodel

    *Extends:* :class:`pybamm.SubModel`
    """

    def __init__(self, set_of_parameters):
        super().__init__(set_of_parameters)

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
        epsilon = variables["Porosity"]
        deps_n_dt = sum(rxn["neg"]["deps_dt"] for rxn in reactions.values())
        deps_p_dt = sum(rxn["pos"]["deps_dt"] for rxn in reactions.values())

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

    def get_variables(self, c, N, species="oxygen"):
        """ See :meth:`pybamm.electrolyte_diffusion.StefanMaxwell.get_variables`. """
        return super().get_variables(c, N, species)
