#
# Lead acid base model class
#

import pybamm


class BaseModel(pybamm.BaseBatteryModel):
    """
    Overwrites default parameters from Base Model with default parameters for
    lead-acid models

    Parameters
    ----------
    options : dict-like, optional
        A dictionary of options to be passed to the model. If this is a dict (and not
        a subtype of dict), it will be processed by :class:`pybamm.BatteryModelOptions`
        to ensure that the options are valid. If this is a subtype of dict, it is
        assumed that the options have already been processed and are valid. This allows
        for the use of custom options classes. The default options are given by
        :class:`pybamm.BatteryModelOptions`.
    name : str, optional
        The name of the model. The default is "Unnamed battery model".
    build : bool, optional
        Whether to build the model on instantiation. Default is True. Setting this
        option to False allows users to change any number of the submodels before
        building the complete model (submodels cannot be changed after the model is
        built).

    **Extends:** :class:`pybamm.BaseBatteryModel`

    """

    def __init__(self, options=None, name="Unnamed lead-acid model", build=False):
        options = options or {}
        # Specify that there are no particles in lead-acid
        options["particle shape"] = "no particles"
        super().__init__(options, name)
        self.param = pybamm.LeadAcidParameters()

        # Default timescale
        self._timescale = self.param.timescale

        # Set default length scales
        self._length_scales = {
            "negative electrode": self.param.L_x,
            "separator": self.param.L_x,
            "positive electrode": self.param.L_x,
            "current collector y": self.param.L_z,
            "current collector z": self.param.L_z,
        }

        self.set_standard_output_variables()

    @property
    def default_parameter_values(self):
        return pybamm.ParameterValues("Sulzer2019")

    @property
    def default_geometry(self):
        return pybamm.battery_geometry(include_particles=False, options=self.options)

    @property
    def default_var_pts(self):
        # Choose points that give uniform grid for the standard parameter values
        return {"x_n": 25, "x_s": 41, "x_p": 34, "y": 10, "z": 10}

    @property
    def default_quick_plot_variables(self):
        return [
            "Interfacial current density [A.m-2]",
            "Electrolyte concentration [mol.m-3]",
            "Current [A]",
            "Porosity",
            "Electrolyte potential [V]",
            "Terminal voltage [V]",
        ]

    def set_soc_variables(self):
        """Set variables relating to the state of charge."""
        # State of Charge defined as function of dimensionless electrolyte concentration
        z = pybamm.standard_spatial_vars.z
        soc = (
            pybamm.Integral(self.variables["X-averaged electrolyte concentration"], z)
            * 100
        )
        self.variables.update({"State of Charge": soc, "Depth of Discharge": 100 - soc})

        # Fractional charge input
        if "Fractional Charge Input" not in self.variables:
            fci = pybamm.Variable("Fractional Charge Input", domain="current collector")
            self.variables["Fractional Charge Input"] = fci
            self.rhs[fci] = -self.variables["Total current density"] * 100
            self.initial_conditions[fci] = self.param.q_init * 100

    def set_open_circuit_potential_submodel(self):
        for domain in ["negative", "positive"]:
            self.submodels[
                f"{domain} open circuit potential"
            ] = pybamm.open_circuit_potential.SingleOpenCircuitPotential(
                self.param, domain, "lead-acid main", self.options, "primary"
            )
            self.submodels[
                f"{domain} oxygen open circuit potential"
            ] = pybamm.open_circuit_potential.SingleOpenCircuitPotential(
                self.param, domain, "lead-acid oxygen", self.options, "primary"
            )

    def set_active_material_submodel(self):
        for domain in ["negative", "positive"]:
            self.submodels[
                f"{domain} active material"
            ] = pybamm.active_material.Constant(
                self.param, domain, self.options, "primary"
            )

    def set_sei_submodel(self):
        self.submodels["sei"] = pybamm.sei.NoSEI(self.param, self.options)

    def set_lithium_plating_submodel(self):
        self.submodels["lithium plating"] = pybamm.lithium_plating.NoPlating(self.param)

    def set_total_interface_submodel(self):
        self.submodels["total interface"] = pybamm.interface.TotalInterfacialCurrent(
            self.param, "lead-acid", self.options
        )
