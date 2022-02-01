#
# Base battery model class
#

import pybamm
import numbers


class BatteryModelOptions(pybamm.FuzzyDict):
    """
    Attributes
    ----------

    options: dict
        A dictionary of options to be passed to the model. The options that can
        be set are listed below. Note that not all of the options are compatible with
        each other and with all of the models implemented in PyBaMM. Each option is
        optional and takes a default value if not provided.
        In general, the option provided must be a string, but there are some cases
        where a 2-tuple of strings can be provided instead to indicate a different
        option for the negative and positive electrodes.

            * "cell geometry" : str
                Sets the geometry of the cell. Can be "pouch" (default) or
                "arbitrary". The arbitrary geometry option solves a 1D electrochemical
                model with prescribed cell volume and cross-sectional area, and
                (if thermal effects are included) solves a lumped thermal model
                with prescribed surface area for cooling.
            * "convection" : str
                Whether to include the effects of convection in the model. Can be
                "none" (default), "uniform transverse" or "full transverse".
                Must be "none" for lithium-ion models.
            * "current collector" : str
                Sets the current collector model to use. Can be "uniform" (default),
                "potential pair" or "potential pair quite conductive".
            * "dimensionality" : int
                Sets the dimension of the current collector problem. Can be 0
                (default), 1 or 2.
            * "electrolyte conductivity" : str
                Can be "default" (default), "full", "leading order", "composite" or
                "integrated".
            * "external submodels" : list
                A list of the submodels that you would like to supply an external
                variable for instead of solving in PyBaMM. The entries of the lists
                are strings that correspond to the submodel names in the keys
                of `self.submodels`.
            * "hydrolysis" : str
                Whether to include hydrolysis in the model. Only implemented for
                lead-acid models. Can be "false" (default) or "true". If "true", then
                "surface form" cannot be 'false'.
            * "intercalation kinetics" : str
                Model for intercalation kinetics. Can be "symmetric Butler-Volmer"
                (default), "asymmetric Butler-Volmer", "linear", "Marcus", or
                "Marcus-Hush-Chidsey" (which uses the asymptotic form from Zeng 2014).
                A 2-tuple can be provided for different behaviour in negative and
                positive electrodes.
            * "interface utilisation": str
                Can be "full" (default), "constant", or "current-driven".
            * "lithium plating" : str
                Sets the model for lithium plating. Can be "none" (default),
                "reversible" or "irreversible".
            * "loss of active material" : str
                Sets the model for loss of active material. Can be "none" (default),
                "stress-driven", "reaction-driven", or "stress and reaction-driven".
                A 2-tuple can be provided for different behaviour in negative and
                positive electrodes.
            * "operating mode" : str
                Sets the operating mode for the model. Can be "current" (default),
                "voltage" or "power". Alternatively, the operating mode can be
                controlled with an arbitrary function by passing the function directly
                as the option. In this case the function must define the residual of
                an algebraic equation. The applied current will be solved for such
                that the algebraic constraint is satisfied.
            * "particle" : str
                Sets the submodel to use to describe behaviour within the particle.
                Can be "Fickian diffusion" (default), "uniform profile",
                "quadratic profile", or "quartic profile".
            * "particle shape" : str
                Sets the model shape of the electrode particles. This is used to
                calculate the surface area to volume ratio. Can be "spherical"
                (default), or "no particles".
            * "particle size" : str
                Sets the model to include a single active particle size or a
                distribution of sizes at any macroscale location. Can be "single"
                (default) or "distribution". Option applies to both electrodes.
            * "particle mechanics" : str
                Sets the model to account for mechanical effects such as particle
                swelling and cracking. Can be "none" (default), "swelling only",
                or "swelling and cracking".
                A 2-tuple can be provided for different behaviour in negative and
                positive electrodes.
            * "SEI" : str
                Set the SEI submodel to be used. Options are:

                - "none": :class:`pybamm.sei.NoSEI` (no SEI growth)
                - "constant": :class:`pybamm.sei.Constant` (constant SEI thickness)
                - "reaction limited", "solvent-diffusion limited",\
                    "electron-migration limited", "interstitial-diffusion limited", \
                    or "ec reaction limited": :class:`pybamm.sei.SEIGrowth`
            * "SEI film resistance" : str
                Set the submodel for additional term in the overpotential due to SEI.
                The default value is "none" if the "SEI" option is "none", and
                "distributed" otherwise. This is because the "distributed" model is more
                complex than the model with no additional resistance, which adds
                unnecessary complexity if there is no SEI in the first place

                - "none": no additional resistance\

                    .. math::
                        \\eta_r = \\frac{F}{RT} * (\\phi_s - \\phi_e - U)

                - "distributed": properly included additional resistance term\

                    .. math::
                        \\eta_r = \\frac{F}{RT}
                        * (\\phi_s - \\phi_e - U - R_{sei} * L_{sei} * j)

                - "average": constant additional resistance term (approximation to the \
                    true model). This model can give similar results to the \
                    "distributed" case without needing to make j an algebraic state\

                    .. math::
                        \\eta_r = \\frac{F}{RT}
                        * (\\phi_s - \\phi_e - U - R_{sei} * L_{sei} * \\frac{I}{aL})
            * "SEI porosity change" : str
                Whether to include porosity change due to SEI formation, can be "false"
                (default) or "true".
            * "stress-induced diffusion" : str
                Whether to include stress-induced diffusion, can be "false" or "true".
                The default is "false" if "particle mechanics" is "none" and "true"
                otherwise. A 2-tuple can be provided for different behaviour in negative
                and positive electrodes.
            * "surface form" : str
                Whether to use the surface formulation of the problem. Can be "false"
                (default), "differential" or "algebraic".
            * "thermal" : str
                Sets the thermal model to use. Can be "isothermal" (default), "lumped",
                "x-lumped", or "x-full".
            * "timescale" : str or number
                Sets the timescale of the model. If "default", the discharge timescale,
                as defined by other parameters, is used. Otherwise, the number is used.
            * "total interfacial current density as a state" : str
                Whether to make a state for the total interfacial current density and
                solve an algebraic equation for it. Default is "false", unless "SEI film
                resistance" is distributed in which case it is automatically set to
                "true".
            * "working electrode": str
                Which electrode(s) intercalates and which is counter. If "both"
                (default), the model is a standard battery. Otherwise can be "negative"
                or "positive" to indicate a half-cell model.

    **Extends:** :class:`dict`
    """

    def __init__(self, extra_options):
        self.possible_options = {
            "cell geometry": ["arbitrary", "pouch"],
            "convection": ["none", "uniform transverse", "full transverse"],
            "current collector": [
                "uniform",
                "potential pair",
                "potential pair quite conductive",
            ],
            "dimensionality": [0, 1, 2],
            "electrolyte conductivity": [
                "default",
                "full",
                "leading order",
                "composite",
                "integrated",
            ],
            "hydrolysis": ["false", "true"],
            "intercalation kinetics": [
                "symmetric Butler-Volmer",
                "asymmetric Butler-Volmer",
                "linear",
                "Marcus",
                "Marcus-Hush-Chidsey",
            ],
            "interface utilisation": ["full", "constant", "current-driven"],
            "lithium plating": ["none", "reversible", "irreversible"],
            "lithium plating porosity change": ["false", "true"],
            "loss of active material": [
                "none",
                "stress-driven",
                "reaction-driven",
                "stress and reaction-driven",
            ],
            "operating mode": ["current", "voltage", "power", "CCCV"],
            "particle": [
                "Fickian diffusion",
                "fast diffusion",
                "uniform profile",
                "quadratic profile",
                "quartic profile",
            ],
            "particle mechanics": ["none", "swelling only", "swelling and cracking"],
            "particle shape": ["spherical", "no particles"],
            "particle size": ["single", "distribution"],
            "SEI": [
                "none",
                "constant",
                "reaction limited",
                "solvent-diffusion limited",
                "electron-migration limited",
                "interstitial-diffusion limited",
                "ec reaction limited",
            ],
            "SEI film resistance": ["none", "distributed", "average"],
            "SEI porosity change": ["false", "true"],
            "stress-induced diffusion": ["false", "true"],
            "surface form": ["false", "differential", "algebraic"],
            "thermal": ["isothermal", "lumped", "x-lumped", "x-full"],
            "total interfacial current density as a state": ["false", "true"],
            "working electrode": ["both", "negative", "positive"],
        }

        default_options = {
            name: options[0] for name, options in self.possible_options.items()
        }
        default_options["external submodels"] = []
        default_options["timescale"] = "default"

        # Change the default for cell geometry based on which thermal option is provided
        extra_options = extra_options or {}
        # return "none" if option not given
        thermal_option = extra_options.get("thermal", "none")
        if thermal_option in ["none", "isothermal", "lumped"]:
            default_options["cell geometry"] = "arbitrary"
        else:
            default_options["cell geometry"] = "pouch"
        # The "cell geometry" option will still be overridden by extra_options if
        # provided

        # Change the default for SEI film resistance based on which SEI option is
        # provided
        # return "none" if option not given
        sei_option = extra_options.get("SEI", "none")
        if sei_option == "none":
            default_options["SEI film resistance"] = "none"
        else:
            default_options["SEI film resistance"] = "distributed"
        # The "SEI film resistance" option will still be overridden by extra_options if
        # provided

        # Change the default for particle mechanics based on which LAM option is
        # provided
        # return "none" if option not given
        lam_option = extra_options.get("loss of active material", "none")
        if "stress-driven" in lam_option or "stress and reaction-driven" in lam_option:
            default_options["particle mechanics"] = "swelling only"
        else:
            default_options["particle mechanics"] = "none"
        # The "particle mechanics" option will still be overridden by extra_options if
        # provided

        # Change the default for stress-induced diffusion based on which particle
        # mechanics option is provided. If the user doesn't supply a particle mechanics
        # option set the default stress-induced diffusion option based on the default
        # particle mechanics option which may change depending on other options
        # (e.g. for stress-driven LAM the default mechanics option is "swelling only")
        mechanics_option = extra_options.get("particle mechanics", "none")
        if (
            mechanics_option == "none"
            and default_options["particle mechanics"] == "none"
        ):
            default_options["stress-induced diffusion"] = "false"
        else:
            default_options["stress-induced diffusion"] = "true"
        # The "stress-induced diffusion" option will still be overridden by
        # extra_options if provided

        options = pybamm.FuzzyDict(default_options)
        # any extra options overwrite the default options
        for name, opt in extra_options.items():
            if name in default_options:
                options[name] = opt
            else:
                if name == "particle cracking":
                    raise pybamm.OptionError(
                        "The 'particle cracking' option has been renamed. "
                        "Use 'particle mechanics' instead."
                    )
                else:
                    raise pybamm.OptionError(
                        "Option '{}' not recognised. Best matches are {}".format(
                            name, options.get_best_matches(name)
                        )
                    )

        # If "SEI film resistance" is "distributed" then "total interfacial current
        # density as a state" must be "true"
        if options["SEI film resistance"] == "distributed":
            options["total interfacial current density as a state"] = "true"
            # Check that extra_options did not try to provide a clashing option
            if (
                extra_options.get("total interfacial current density as a state")
                == "false"
            ):
                raise pybamm.OptionError(
                    "If 'sei film resistance' is 'distributed' then 'total interfacial "
                    "current density as a state' must be 'true'"
                )

        # Options not yet compatible with particle-size distributions
        if options["particle size"] == "distribution":
            if options["lithium plating"] != "none":
                raise NotImplementedError(
                    "Lithium plating submodels do not yet support particle-size "
                    "distributions."
                )
            if options["particle"] in ["quadratic profile", "quartic profile"]:
                raise NotImplementedError(
                    "'quadratic' and 'quartic' concentration profiles have not yet "
                    "been implemented for particle-size ditributions"
                )
            if options["particle mechanics"] != "none":
                raise NotImplementedError(
                    "Particle mechanics submodels do not yet support particle-size"
                    " distributions."
                )
            if options["particle shape"] != "spherical":
                raise NotImplementedError(
                    "Particle shape must be 'spherical' for particle-size distribution"
                    " submodels."
                )
            if options["SEI"] != "none":
                raise NotImplementedError(
                    "SEI submodels do not yet support particle-size distributions."
                )
            if options["stress-induced diffusion"] == "true":
                raise NotImplementedError(
                    "stress-induced diffusion cannot yet be included in "
                    "particle-size distributions."
                )
            if options["thermal"] == "x-full":
                raise NotImplementedError(
                    "X-full thermal submodels do not yet support particle-size"
                    " distributions."
                )

        # Renamed options
        if options["particle"] == "fast diffusion":
            raise pybamm.OptionError(
                "The 'fast diffusion' option has been renamed. "
                "Use 'uniform profile' instead."
            )
        if options["SEI porosity change"] in [True, False]:
            raise pybamm.OptionError(
                "SEI porosity change must now be given in string format "
                "('true' or 'false')"
            )

        # Some standard checks to make sure options are compatible
        if options["dimensionality"] == 0:
            if options["current collector"] not in ["uniform"]:
                raise pybamm.OptionError(
                    "current collector model must be uniform in 0D model"
                )
            if options["convection"] == "full transverse":
                raise pybamm.OptionError(
                    "cannot have transverse convection in 0D model"
                )

        if options["thermal"] == "x-full" and options["dimensionality"] != 0:
            n = options["dimensionality"]
            raise pybamm.OptionError(
                f"X-full thermal submodels do not yet support {n}D current collectors"
            )

        if isinstance(options["stress-induced diffusion"], str):
            if (
                options["stress-induced diffusion"] == "true"
                and options["particle mechanics"] == "none"
            ):
                raise pybamm.OptionError(
                    "cannot have stress-induced diffusion without a particle "
                    "mechanics model"
                )

        # Check options are valid
        for option, value in options.items():
            if option == "external submodels" or option == "working electrode":
                pass
            else:
                if isinstance(value, str) or option in [
                    "dimensionality",
                    "operating mode",
                    "timescale",
                ]:  # some options accept non-strings
                    value = (value,)
                else:
                    if not (
                        (
                            option
                            in [
                                "intercalation kinetics",
                                "interface utilisation",
                                "loss of active material",
                                "particle mechanics",
                                "particle",
                                "stress-induced diffusion",
                            ]
                            and isinstance(value, tuple)
                            and len(value) == 2
                        )
                    ):
                        # more possible options that can take 2-tuples to be added
                        # as they come
                        raise pybamm.OptionError(
                            f"\n'{value}' is not recognized in option '{option}'. "
                            "Values must be strings or (in some cases) "
                            "2-tuples of strings"
                        )
                for val in value:
                    if option == "timescale":
                        if not (val == "default" or isinstance(val, numbers.Number)):
                            raise pybamm.OptionError(
                                "'timescale' option must be either 'default' "
                                "or a number"
                            )
                    elif val not in self.possible_options[option]:
                        if not (option == "operating mode" and callable(val)):
                            raise pybamm.OptionError(
                                f"\n'{val}' is not recognized in option '{option}'. "
                                f"Possible values are {self.possible_options[option]}"
                            )

        super().__init__(options.items())

    def print_options(self):
        """
        Print the possible options with the ones currently selected
        """
        for key, value in self.items():
            if key in self.possible_options.keys():
                print(f"{key!r}: {value!r} (possible: {self.possible_options[key]!r})")
            else:
                print(f"{key!r}: {value!r}")

    def print_detailed_options(self):
        """
        Print the docstring for Options
        """
        print(self.__doc__)

    @property
    def negative(self):
        "Returns the options for the negative electrode"
        # index 0 in a 2-tuple for the negative electrode
        return BatteryModelDomainOptions(self.items(), 0)

    @property
    def positive(self):
        "Returns the options for the positive electrode"
        # index 1 in a 2-tuple for the positive electrode
        return BatteryModelDomainOptions(self.items(), 1)


class BatteryModelDomainOptions(dict):
    def __init__(self, dict_items, index):
        super().__init__(dict_items)
        self.index = index

    def __getitem__(self, key):
        options = super().__getitem__(key)
        if isinstance(options, str):
            return options
        else:
            # 2-tuple, first is negative domain, second is positive domain
            return options[self.index]


class BaseBatteryModel(pybamm.BaseModel):
    """
    Base model class with some default settings and required variables
    **Extends:** :class:`pybamm.BaseModel`
    """

    def __init__(self, options=None, name="Unnamed battery model"):
        super().__init__(name)
        self.options = options
        self.submodels = {}
        self._built = False
        self._built_fundamental_and_external = False

    @pybamm.BaseModel.timescale.setter
    def timescale(self, value):
        """Set the timescale"""
        raise NotImplementedError(
            "Timescale cannot be directly overwritten for this model. "
            "Pass a timescale to the 'timescale' option instead."
        )

    @pybamm.BaseModel.length_scales.setter
    def length_scales(self, value):
        """Set the length scales"""
        raise NotImplementedError(
            "Length scales cannot be directly overwritten for this model. "
        )

    @property
    def default_geometry(self):
        return pybamm.battery_geometry(
            options=self.options,
            current_collector_dimension=self.options["dimensionality"],
        )

    @property
    def default_var_pts(self):
        base_var_pts = {
            "x_n": 20,
            "x_s": 20,
            "x_p": 20,
            "r_n": 20,
            "r_p": 20,
            "y": 10,
            "z": 10,
            "R_n": 30,
            "R_p": 30,
        }
        # Reduce the default points for 2D current collectors
        if self.options["dimensionality"] == 2:
            base_var_pts.update({"x_n": 10, "x_s": 10, "x_p": 10})
        return base_var_pts

    @property
    def default_submesh_types(self):
        base_submeshes = {
            "negative electrode": pybamm.Uniform1DSubMesh,
            "separator": pybamm.Uniform1DSubMesh,
            "positive electrode": pybamm.Uniform1DSubMesh,
            "negative particle": pybamm.Uniform1DSubMesh,
            "positive particle": pybamm.Uniform1DSubMesh,
            "negative particle size": pybamm.Uniform1DSubMesh,
            "positive particle size": pybamm.Uniform1DSubMesh,
        }
        if self.options["dimensionality"] == 0:
            base_submeshes["current collector"] = pybamm.SubMesh0D
        elif self.options["dimensionality"] == 1:
            base_submeshes["current collector"] = pybamm.Uniform1DSubMesh

        elif self.options["dimensionality"] == 2:
            base_submeshes["current collector"] = pybamm.ScikitUniform2DSubMesh
        return base_submeshes

    @property
    def default_spatial_methods(self):
        base_spatial_methods = {
            "macroscale": pybamm.FiniteVolume(),
            "negative particle": pybamm.FiniteVolume(),
            "positive particle": pybamm.FiniteVolume(),
            "negative particle size": pybamm.FiniteVolume(),
            "positive particle size": pybamm.FiniteVolume(),
        }
        if self.options["dimensionality"] == 0:
            # 0D submesh - use base spatial method
            base_spatial_methods[
                "current collector"
            ] = pybamm.ZeroDimensionalSpatialMethod()
        elif self.options["dimensionality"] == 1:
            base_spatial_methods["current collector"] = pybamm.FiniteVolume()
        elif self.options["dimensionality"] == 2:
            base_spatial_methods["current collector"] = pybamm.ScikitFiniteElement()
        return base_spatial_methods

    @property
    def options(self):
        return self._options

    @options.setter
    def options(self, extra_options):
        options = BatteryModelOptions(extra_options)

        # Options that are incompatible with models
        if isinstance(self, pybamm.lithium_ion.BaseModel):
            if options["convection"] != "none":
                raise pybamm.OptionError(
                    "convection not implemented for lithium-ion models"
                )
            if (
                options["thermal"] in ["x-lumped", "x-full"]
                and options["cell geometry"] != "pouch"
            ):
                raise pybamm.OptionError(
                    options["thermal"] + " model must have pouch geometry."
                )
        if isinstance(self, pybamm.lithium_ion.SPMe):
            if options["electrolyte conductivity"] not in [
                "default",
                "composite",
                "integrated",
            ]:
                raise pybamm.OptionError(
                    "electrolyte conductivity '{}' not suitable for SPMe".format(
                        options["electrolyte conductivity"]
                    )
                )
        if isinstance(self, pybamm.lead_acid.BaseModel):
            if options["thermal"] != "isothermal" and options["dimensionality"] != 0:
                raise pybamm.OptionError(
                    "Lead-acid models can only have thermal "
                    "effects if dimensionality is 0."
                )
            if options["SEI"] != "none" or options["SEI film resistance"] != "none":
                raise pybamm.OptionError("Lead-acid models cannot have SEI formation")
            if options["lithium plating"] != "none":
                raise pybamm.OptionError("Lead-acid models cannot have lithium plating")

        if (
            isinstance(self, (pybamm.lead_acid.LOQS, pybamm.lead_acid.Composite))
            and options["surface form"] == "false"
            and options["hydrolysis"] == "true"
        ):
            raise pybamm.OptionError(
                """must use surface formulation to solve {!s} with hydrolysis
                    """.format(
                    self
                )
            )

        self._options = options

    def set_standard_output_variables(self):
        # Time
        self.variables.update(
            {
                "Time": pybamm.t,
                "Time [s]": pybamm.t * self.timescale,
                "Time [min]": pybamm.t * self.timescale / 60,
                "Time [h]": pybamm.t * self.timescale / 3600,
            }
        )

        # Spatial
        var = pybamm.standard_spatial_vars
        L_x = self.param.L_x
        L_z = self.param.L_z
        self.variables.update(
            {
                "x": var.x,
                "x [m]": var.x * L_x,
                "x_n": var.x_n,
                "x_n [m]": var.x_n * L_x,
                "x_s": var.x_s,
                "x_s [m]": var.x_s * L_x,
                "x_p": var.x_p,
                "x_p [m]": var.x_p * L_x,
            }
        )
        if self.options["dimensionality"] == 1:
            self.variables.update({"z": var.z, "z [m]": var.z * L_z})
        elif self.options["dimensionality"] == 2:
            # Note: both y and z are scaled with L_z
            self.variables.update(
                {"y": var.y, "y [m]": var.y * L_z, "z": var.z, "z [m]": var.z * L_z}
            )

        # Initialize "total reaction" variables
        # These will get populated by the "get_coupled_variables" methods, and then used
        # later by "set_rhs" or "set_algebraic", which ensures that we always have
        # added all the necessary variables by the time the sum is used
        self.variables.update(
            {
                "Sum of electrolyte reaction source terms": 0,
                "Sum of positive electrode electrolyte reaction source terms": 0,
                "Sum of x-averaged positive electrode "
                "electrolyte reaction source terms": 0,
                "Sum of interfacial current densities": 0,
                "Sum of positive electrode interfacial current densities": 0,
                "Sum of x-averaged positive electrode interfacial current densities": 0,
            }
        )
        if not self.half_cell:
            self.variables.update(
                {
                    "Sum of negative electrode electrolyte reaction source terms": 0,
                    "Sum of x-averaged negative electrode "
                    "electrolyte reaction source terms": 0,
                    "Sum of negative electrode interfacial current densities": 0,
                    "Sum of x-averaged negative electrode interfacial current densities"
                    "": 0,
                }
            )

    def build_fundamental_and_external(self):
        # Get the fundamental variables
        for submodel_name, submodel in self.submodels.items():
            pybamm.logger.debug(
                "Getting fundamental variables for {} submodel ({})".format(
                    submodel_name, self.name
                )
            )
            self.variables.update(submodel.get_fundamental_variables())

        # Set the submodels that are external
        for sub in self.options["external submodels"]:
            self.submodels[sub].external = True

        # Set any external variables
        self.external_variables = []
        for submodel_name, submodel in self.submodels.items():
            pybamm.logger.debug(
                "Getting external variables for {} submodel ({})".format(
                    submodel_name, self.name
                )
            )
            external_variables = submodel.get_external_variables()

            self.external_variables += external_variables

        self._built_fundamental_and_external = True

    def build_coupled_variables(self):
        # Note: pybamm will try to get the coupled variables for the submodels in the
        # order they are set by the user. If this fails for a particular submodel,
        # return to it later and try again. If setting coupled variables fails and
        # there are no more submodels to try, raise an error.
        submodels = list(self.submodels.keys())
        count = 0
        # For this part the FuzzyDict of variables is briefly converted back into a
        # normal dictionary for speed with KeyErrors
        self._variables = dict(self._variables)
        while len(submodels) > 0:
            count += 1
            for submodel_name, submodel in self.submodels.items():
                if submodel_name in submodels:
                    pybamm.logger.debug(
                        "Getting coupled variables for {} submodel ({})".format(
                            submodel_name, self.name
                        )
                    )
                    try:
                        self.variables.update(
                            submodel.get_coupled_variables(self.variables)
                        )
                        submodels.remove(submodel_name)
                    except KeyError as key:
                        if len(submodels) == 1 or count == 100:
                            # no more submodels to try
                            raise pybamm.ModelError(
                                "Missing variable for submodel '{}': {}.\n".format(
                                    submodel_name, key
                                )
                                + "Check the selected "
                                "submodels provide all of the required variables."
                            )
                        else:
                            # try setting coupled variables on next loop through
                            pybamm.logger.debug(
                                "Can't find {}, trying other submodels first".format(
                                    key
                                )
                            )
        # Convert variables back into FuzzyDict
        self.variables = pybamm.FuzzyDict(self._variables)

    def build_model_equations(self):
        # Set model equations
        for submodel_name, submodel in self.submodels.items():
            if submodel.external is False:
                pybamm.logger.verbose(
                    "Setting rhs for {} submodel ({})".format(submodel_name, self.name)
                )

                submodel.set_rhs(self.variables)
                pybamm.logger.verbose(
                    "Setting algebraic for {} submodel ({})".format(
                        submodel_name, self.name
                    )
                )

                submodel.set_algebraic(self.variables)
                pybamm.logger.verbose(
                    "Setting boundary conditions for {} submodel ({})".format(
                        submodel_name, self.name
                    )
                )

                submodel.set_boundary_conditions(self.variables)
                pybamm.logger.verbose(
                    "Setting initial conditions for {} submodel ({})".format(
                        submodel_name, self.name
                    )
                )
                submodel.set_initial_conditions(self.variables)
                submodel.set_events(self.variables)
                pybamm.logger.verbose(
                    "Updating {} submodel ({})".format(submodel_name, self.name)
                )
                self.update(submodel)
                self.check_no_repeated_keys()

    def build_model(self):

        # Check if already built
        if self._built:
            raise pybamm.ModelError(
                """Model already built. If you are adding a new submodel, try using
                `model.update` instead."""
            )

        pybamm.logger.info("Start building {}".format(self.name))

        if self._built_fundamental_and_external is False:
            self.build_fundamental_and_external()

        self.build_coupled_variables()

        self.build_model_equations()

        pybamm.logger.debug("Setting voltage variables ({})".format(self.name))
        self.set_voltage_variables()

        pybamm.logger.debug("Setting SoC variables ({})".format(self.name))
        self.set_soc_variables()

        pybamm.logger.debug("Setting degradation variables ({})".format(self.name))
        self.set_degradation_variables()
        self.set_summary_variables()

        self._built = True
        pybamm.logger.info("Finish building {}".format(self.name))

    def new_empty_copy(self):
        """See :meth:`pybamm.BaseModel.new_empty_copy()`"""
        new_model = self.__class__(name=self.name, options=self.options)
        new_model.use_jacobian = self.use_jacobian
        new_model.convert_to_format = self.convert_to_format
        new_model._timescale = self.timescale
        new_model._length_scales = self.length_scales
        return new_model

    @property
    def summary_variables(self):
        return self._summary_variables

    @summary_variables.setter
    def summary_variables(self, value):
        """
        Set summary variables

        Parameters
        ----------
        value : list of strings
            Names of the summary variables. Must all be in self.variables.
        """
        for var in value:
            if var not in self.variables:
                raise KeyError(
                    f"No cycling variable defined for summary variable '{var}'"
                )
        self._summary_variables = value

    def set_summary_variables(self):
        self._summary_variables = []

    def get_intercalation_kinetics(self, domain):
        options = getattr(self.options, domain.lower())
        if options["intercalation kinetics"] == "symmetric Butler-Volmer":
            return pybamm.kinetics.SymmetricButlerVolmer
        elif options["intercalation kinetics"] == "asymmetric Butler-Volmer":
            return pybamm.kinetics.AsymmetricButlerVolmer
        elif options["intercalation kinetics"] == "linear":
            return pybamm.kinetics.Linear
        elif options["intercalation kinetics"] == "Marcus":
            return pybamm.kinetics.Marcus
        elif options["intercalation kinetics"] == "Marcus-Hush-Chidsey":
            return pybamm.kinetics.MarcusHushChidsey

    @property
    def inverse_intercalation_kinetics(self):
        if self.options["intercalation kinetics"] == "symmetric Butler-Volmer":
            return pybamm.kinetics.InverseButlerVolmer
        else:
            raise pybamm.OptionError(
                "Inverse kinetics are only implemented for symmetric Butler-Volmer. "
                "Use option {'surface form': 'algebraic'} to use forward kinetics "
                "instead."
            )

    def set_external_circuit_submodel(self):
        """
        Define how the external circuit defines the boundary conditions for the model,
        e.g. (not necessarily constant-) current, voltage, etc
        """
        if self.options["operating mode"] == "current":
            self.submodels["external circuit"] = pybamm.external_circuit.CurrentControl(
                self.param
            )
        elif self.options["operating mode"] == "voltage":
            self.submodels[
                "external circuit"
            ] = pybamm.external_circuit.VoltageFunctionControl(self.param)
        elif self.options["operating mode"] == "power":
            self.submodels[
                "external circuit"
            ] = pybamm.external_circuit.PowerFunctionControl(self.param)
        elif self.options["operating mode"] == "CCCV":
            self.submodels[
                "external circuit"
            ] = pybamm.external_circuit.CCCVFunctionControl(self.param)
        elif callable(self.options["operating mode"]):
            self.submodels[
                "external circuit"
            ] = pybamm.external_circuit.FunctionControl(
                self.param, self.options["operating mode"]
            )

    def set_tortuosity_submodels(self):
        self.submodels["electrolyte tortuosity"] = pybamm.tortuosity.Bruggeman(
            self.param, "Electrolyte", self.options
        )
        self.submodels["electrode tortuosity"] = pybamm.tortuosity.Bruggeman(
            self.param, "Electrode", self.options
        )

    def set_thermal_submodel(self):

        if self.options["thermal"] == "isothermal":
            thermal_submodel = pybamm.thermal.isothermal.Isothermal(
                self.param, self.options
            )

        elif self.options["thermal"] == "lumped":
            thermal_submodel = pybamm.thermal.Lumped(
                self.param,
                cc_dimension=self.options["dimensionality"],
                geometry=self.options["cell geometry"],
            )

        elif self.options["thermal"] == "x-lumped":
            if self.options["dimensionality"] == 0:
                # With 0D current collectors x-lumped is equivalent to lumped pouch
                thermal_submodel = pybamm.thermal.Lumped(self.param, geometry="pouch")
            elif self.options["dimensionality"] == 1:
                thermal_submodel = pybamm.thermal.pouch_cell.CurrentCollector1D(
                    self.param
                )
            elif self.options["dimensionality"] == 2:
                thermal_submodel = pybamm.thermal.pouch_cell.CurrentCollector2D(
                    self.param
                )
        elif self.options["thermal"] == "x-full":
            if self.options["dimensionality"] == 0:
                thermal_submodel = pybamm.thermal.OneDimensionalX(self.param)

        self.submodels["thermal"] = thermal_submodel

    def set_current_collector_submodel(self):

        if self.options["current collector"] in ["uniform"]:
            submodel = pybamm.current_collector.Uniform(self.param)
        elif self.options["current collector"] == "potential pair":
            if self.options["dimensionality"] == 1:
                submodel = pybamm.current_collector.PotentialPair1plus1D(self.param)
            elif self.options["dimensionality"] == 2:
                submodel = pybamm.current_collector.PotentialPair2plus1D(self.param)
        self.submodels["current collector"] = submodel

    def set_interface_utilisation_submodel(self):
        if self.half_cell:
            domains = ["Counter", "Positive"]
        else:
            domains = ["Negative", "Positive"]
        for domain in domains:
            name = domain.lower() + " interface utilisation"
            if domain == "Counter":
                domain = "Negative"
            util = getattr(self.options, domain.lower())["interface utilisation"]
            if util == "full":
                self.submodels[name] = pybamm.interface_utilisation.Full(
                    self.param, domain, self.options
                )
            elif util == "constant":
                self.submodels[name] = pybamm.interface_utilisation.Constant(
                    self.param, domain, self.options
                )
            elif util == "current-driven":
                if self.half_cell and domain == "Negative":
                    reaction_loc = "interface"
                elif self.x_average:
                    reaction_loc = "x-average"
                else:
                    reaction_loc = "full electrode"
                self.submodels[name] = pybamm.interface_utilisation.CurrentDriven(
                    self.param, domain, self.options, reaction_loc
                )

    def set_voltage_variables(self):

        ocp_n = self.variables["Negative electrode open circuit potential"]
        ocp_p = self.variables["Positive electrode open circuit potential"]
        ocp_n_av = self.variables[
            "X-averaged negative electrode open circuit potential"
        ]
        ocp_p_av = self.variables[
            "X-averaged positive electrode open circuit potential"
        ]

        ocp_n_dim = self.variables["Negative electrode open circuit potential [V]"]
        ocp_p_dim = self.variables["Positive electrode open circuit potential [V]"]
        ocp_n_av_dim = self.variables[
            "X-averaged negative electrode open circuit potential [V]"
        ]
        ocp_p_av_dim = self.variables[
            "X-averaged positive electrode open circuit potential [V]"
        ]

        ocp_n_left = pybamm.boundary_value(ocp_n, "left")
        ocp_n_left_dim = pybamm.boundary_value(ocp_n_dim, "left")
        ocp_p_right = pybamm.boundary_value(ocp_p, "right")
        ocp_p_right_dim = pybamm.boundary_value(ocp_p_dim, "right")

        ocv_av = ocp_p_av - ocp_n_av
        ocv_av_dim = ocp_p_av_dim - ocp_n_av_dim
        ocv = ocp_p_right - ocp_n_left
        ocv_dim = ocp_p_right_dim - ocp_n_left_dim

        # overpotentials
        if self.half_cell:
            eta_r_n_av = self.variables[
                "Lithium metal interface reaction overpotential"
            ]
            eta_r_n_av_dim = self.variables[
                "Lithium metal interface reaction overpotential [V]"
            ]
        else:
            eta_r_n_av = self.variables[
                "X-averaged negative electrode reaction overpotential"
            ]
            eta_r_n_av_dim = self.variables[
                "X-averaged negative electrode reaction overpotential [V]"
            ]
        eta_r_p_av = self.variables[
            "X-averaged positive electrode reaction overpotential"
        ]
        eta_r_p_av_dim = self.variables[
            "X-averaged positive electrode reaction overpotential [V]"
        ]

        delta_phi_s_n_av = self.variables["X-averaged negative electrode ohmic losses"]
        delta_phi_s_n_av_dim = self.variables[
            "X-averaged negative electrode ohmic losses [V]"
        ]
        delta_phi_s_p_av = self.variables["X-averaged positive electrode ohmic losses"]
        delta_phi_s_p_av_dim = self.variables[
            "X-averaged positive electrode ohmic losses [V]"
        ]

        delta_phi_s_av = delta_phi_s_p_av - delta_phi_s_n_av
        delta_phi_s_av_dim = delta_phi_s_p_av_dim - delta_phi_s_n_av_dim

        eta_r_av = eta_r_p_av - eta_r_n_av
        eta_r_av_dim = eta_r_p_av_dim - eta_r_n_av_dim

        # SEI film overpotential
        if self.half_cell:
            eta_sei_av = self.variables["SEI film overpotential"]
            eta_sei_av_dim = self.variables["SEI film overpotential [V]"]
        else:
            eta_sei_av = self.variables["X-averaged SEI film overpotential"]
            eta_sei_av_dim = self.variables["X-averaged SEI film overpotential [V]"]

        # TODO: add current collector losses to the voltage in 3D

        self.variables.update(
            {
                "X-averaged open circuit voltage": ocv_av,
                "Measured open circuit voltage": ocv,
                "X-averaged open circuit voltage [V]": ocv_av_dim,
                "Measured open circuit voltage [V]": ocv_dim,
                "X-averaged reaction overpotential": eta_r_av,
                "X-averaged reaction overpotential [V]": eta_r_av_dim,
                "X-averaged SEI film overpotential": eta_sei_av,
                "X-averaged SEI film overpotential [V]": eta_sei_av_dim,
                "X-averaged solid phase ohmic losses": delta_phi_s_av,
                "X-averaged solid phase ohmic losses [V]": delta_phi_s_av_dim,
            }
        )

        # Battery-wide variables
        V = self.variables["Terminal voltage"]
        V_dim = self.variables["Terminal voltage [V]"]
        eta_e_av_dim = self.variables["X-averaged electrolyte ohmic losses [V]"]
        eta_c_av_dim = self.variables["X-averaged concentration overpotential [V]"]
        num_cells = pybamm.Parameter(
            "Number of cells connected in series to make a battery"
        )
        self.variables.update(
            {
                "X-averaged battery open circuit voltage [V]": ocv_av_dim * num_cells,
                "Measured battery open circuit voltage [V]": ocv_dim * num_cells,
                "X-averaged battery reaction overpotential [V]": eta_r_av_dim
                * num_cells,
                "X-averaged battery solid phase ohmic losses [V]": delta_phi_s_av_dim
                * num_cells,
                "X-averaged battery electrolyte ohmic losses [V]": eta_e_av_dim
                * num_cells,
                "X-averaged battery concentration overpotential [V]": eta_c_av_dim
                * num_cells,
                "Battery voltage [V]": V_dim * num_cells,
            }
        )
        # Variables for calculating the equivalent circuit model (ECM) resistance
        # Need to compare OCV to initial value to capture this as an overpotential
        ocv_init = self.param.U_p_init - self.param.U_n_init
        ocv_init_dim = (
            self.param.U_p_ref
            - self.param.U_n_ref
            + self.param.potential_scale * ocv_init
        )
        eta_ocv = ocv - ocv_init
        eta_ocv_dim = ocv_dim - ocv_init_dim
        # Current collector current density for working out euiqvalent resistance
        # based on Ohm's Law
        i_cc = self.variables["Current collector current density"]
        i_cc_dim = self.variables["Current collector current density [A.m-2]"]
        # ECM overvoltage is OCV minus terminal voltage
        v_ecm = ocv - V
        v_ecm_dim = ocv_dim - V_dim
        # Current collector area for turning resistivity into resistance
        A_cc = self.param.A_cc

        # Hack to avoid division by zero if i_cc is exactly zero
        # If i_cc is zero, i_cc_not_zero becomes 1. But multiplying by sign(i_cc) makes
        # the local resistance 'zero' (really, it's not defined when i_cc is zero)
        i_cc_not_zero = ((i_cc > 0) + (i_cc < 0)) * i_cc + (i_cc >= 0) * (i_cc <= 0)
        i_cc_dim_not_zero = ((i_cc_dim > 0) + (i_cc_dim < 0)) * i_cc_dim + (
            i_cc_dim >= 0
        ) * (i_cc_dim <= 0)

        self.variables.update(
            {
                "Change in measured open circuit voltage": eta_ocv,
                "Change in measured open circuit voltage [V]": eta_ocv_dim,
                "Local ECM resistance": pybamm.sign(i_cc)
                * v_ecm
                / (i_cc_not_zero * A_cc),
                "Local ECM resistance [Ohm]": pybamm.sign(i_cc)
                * v_ecm_dim
                / (i_cc_dim_not_zero * A_cc),
            }
        )

        # Cut-off voltage
        self.events.append(
            pybamm.Event(
                "Minimum voltage",
                V - self.param.voltage_low_cut,
                pybamm.EventType.TERMINATION,
            )
        )
        self.events.append(
            pybamm.Event(
                "Maximum voltage",
                V - self.param.voltage_high_cut,
                pybamm.EventType.TERMINATION,
            )
        )

        # Cut-off open-circuit voltage (for event switch with casadi 'fast with events'
        # mode)
        # A tolerance of ~1 is sufficiently small since the dimensionless voltage is
        # scaled with the thermal voltage (0.025V) and hence has a range of around 60
        tol = 5
        self.events.append(
            pybamm.Event(
                "Minimum voltage switch",
                V - (self.param.voltage_low_cut - tol),
                pybamm.EventType.SWITCH,
            )
        )
        self.events.append(
            pybamm.Event(
                "Maximum voltage switch",
                V - (self.param.voltage_high_cut + tol),
                pybamm.EventType.SWITCH,
            )
        )

        # Power
        I_dim = self.variables["Current [A]"]
        self.variables.update({"Terminal power [W]": I_dim * V_dim})

    def set_degradation_variables(self):
        """
        Set variables that quantify degradation.
        This function is overriden by the base battery models
        """
        pass

    def set_soc_variables(self):
        """
        Set variables relating to the state of charge.
        This function is overriden by the base battery models
        """
        pass

    def process_parameters_and_discretise(self, symbol, parameter_values, disc):
        """
        Process parameters and discretise a symbol using supplied parameter values
        and discretisation. Note: care should be taken if using spatial operators
        on dimensional symbols. Operators in pybamm are written in non-dimensional
        form, so may need to be scaled by the appropriate length scale. It is
        recommended to use this method on non-dimensional symbols.

        Parameters
        ----------
        symbol : :class:`pybamm.Symbol`
            Symbol to be processed
        parameter_values : :class:`pybamm.ParameterValues`
            The parameter values to use during processing
        disc : :class:`pybamm.Discretisation`
            The discrisation to use

        Returns
        -------
        :class:`pybamm.Symbol`
            Processed symbol
        """
        # Set y slices
        if disc.y_slices == {}:
            variables = list(self.rhs.keys()) + list(self.algebraic.keys())
            disc.set_variable_slices(variables)

        # Set boundary conditions (also requires setting parameter values)
        if disc.bcs == {}:
            self.boundary_conditions = parameter_values.process_boundary_conditions(
                self
            )
            disc.bcs = disc.process_boundary_conditions(self)

        # Process
        param_symbol = parameter_values.process_symbol(symbol)
        disc_symbol = disc.process_symbol(param_symbol)

        return disc_symbol
