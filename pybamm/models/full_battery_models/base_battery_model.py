#
# Base battery model class
#

import pybamm
from functools import cached_property


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

            * "calculate discharge energy": str
                Whether to calculate the discharge energy, throughput energy and
                throughput capacity in addition to discharge capacity. Must be one of
                "true" or "false". "false" is the default, since calculating discharge
                energy can be computationally expensive for simple models like SPM.
            * "cell geometry" : str
                Sets the geometry of the cell. Can be "pouch" (default) or
                "arbitrary". The arbitrary geometry option solves a 1D electrochemical
                model with prescribed cell volume and cross-sectional area, and
                (if thermal effects are included) solves a lumped thermal model
                with prescribed surface area for cooling.
            * "calculate heat source for isothermal models" : str
                Whether to calculate the heat source terms during isothermal operation.
                Can be "true" or "false". If "false", the heat source terms are set
                to zero. Default is "false" since this option may require additional
                parameters not needed by the electrochemical model.
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
                "reversible", "partially reversible", or "irreversible".
            * "lithium plating porosity change" : str
                Whether to include porosity change due to lithium plating, can be
                "false" (default) or "true".
            * "loss of active material" : str
                Sets the model for loss of active material. Can be "none" (default),
                "stress-driven", "reaction-driven", or "stress and reaction-driven".
                A 2-tuple can be provided for different behaviour in negative and
                positive electrodes.
            * "operating mode" : str
                Sets the operating mode for the model. This determines how the current
                is set. Can be:

                - "current" (default) : the current is explicity supplied
                - "voltage"/"power"/"resistance" : solve an algebraic equation for \
                    current such that voltage/power/resistance is correct
                - "differential power"/"differential resistance" : solve a \
                    differential equation for the power or resistance
                - "explicit power"/"explicit resistance" : current is defined in terms \
                    of the voltage such that power/resistance is correct
                - "CCCV": a special implementation of the common constant-current \
                    constant-voltage charging protocol, via an ODE for the current
                - callable : if a callable is given as this option, the function \
                    defines the residual of an algebraic equation. The applied current \
                    will be solved for such that the algebraic constraint is satisfied.
            * "particle" : str
                Sets the submodel to use to describe behaviour within the particle.
                Can be "Fickian diffusion" (default), "uniform profile",
                "quadratic profile", or "quartic profile". A 2-tuple can be provided 
                for different behaviour in negative and positive electrodes.
            * "particle mechanics" : str
                Sets the model to account for mechanical effects such as particle
                swelling and cracking. Can be "none" (default), "swelling only",
                or "swelling and cracking".
                A 2-tuple can be provided for different behaviour in negative and
                positive electrodes.
            * "particle phases": str
                Number of phases present in the electrode. A 2-tuple can be provided for
                different behaviour in negative and positive electrodes.
                For example, set to ("2", "1") for a negative electrode with 2 phases,
                e.g. graphite and silicon.
            * "particle shape" : str
                Sets the model shape of the electrode particles. This is used to
                calculate the surface area to volume ratio. Can be "spherical"
                (default), or "no particles".
            * "particle size" : str
                Sets the model to include a single active particle size or a
                distribution of sizes at any macroscale location. Can be "single"
                (default) or "distribution". Option applies to both electrodes.
            * "SEI" : str
                Set the SEI submodel to be used. Options are:

                - "none": :class:`pybamm.sei.NoSEI` (no SEI growth)
                - "constant": :class:`pybamm.sei.Constant` (constant SEI thickness)
                - "reaction limited", "reaction limited (asymmetric)", \
                    "solvent-diffusion limited", "electron-migration limited", \
                    "interstitial-diffusion limited", "ec reaction limited" \
                    or "ec reaction limited (asymmetric)": :class:`pybamm.sei.SEIGrowth`
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
            * "SEI on cracks" : str
                Whether to include SEI growth on particle cracks, can be "false"
                (default) or "true".
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
            * "total interfacial current density as a state" : str
                Whether to make a state for the total interfacial current density and
                solve an algebraic equation for it. Default is "false", unless "SEI film
                resistance" is distributed in which case it is automatically set to
                "true".
            * "working electrode": str
                Which electrode(s) intercalates and which is counter. If "both"
                (default), the model is a standard battery. Otherwise can be "negative"
                or "positive" to indicate a half-cell model.
            * "x-average side reactions": str
                Whether to average the side reactions (SEI growth, lithium plating and
                the respective porosity change) over the x-axis in Single Particle
                Models, can be "false" or "true". Default is "false" for SPMe and
                "true" for SPM.
    """

    def __init__(self, extra_options):
        self.possible_options = {
            "calculate discharge energy": ["false", "true"],
            "calculate heat source for isothermal models": ["false", "true"],
            "cell geometry": ["arbitrary", "pouch"],
            "contact resistance": ["false", "true"],
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
            "lithium plating": [
                "none",
                "reversible",
                "partially reversible",
                "irreversible",
            ],
            "lithium plating porosity change": ["false", "true"],
            "loss of active material": [
                "none",
                "stress-driven",
                "reaction-driven",
                "stress and reaction-driven",
            ],
            "open-circuit potential": ["single", "current sigmoid"],
            "operating mode": [
                "current",
                "voltage",
                "power",
                "differential power",
                "explicit power",
                "resistance",
                "differential resistance",
                "explicit resistance",
                "CCCV",
            ],
            "particle": [
                "Fickian diffusion",
                "fast diffusion",
                "uniform profile",
                "quadratic profile",
                "quartic profile",
            ],
            "particle mechanics": ["none", "swelling only", "swelling and cracking"],
            "particle phases": ["1", "2"],
            "particle shape": ["spherical", "no particles"],
            "particle size": ["single", "distribution"],
            "SEI": [
                "none",
                "constant",
                "reaction limited",
                "reaction limited (asymmetric)",
                "solvent-diffusion limited",
                "electron-migration limited",
                "interstitial-diffusion limited",
                "ec reaction limited",
                "ec reaction limited (asymmetric)",
            ],
            "SEI film resistance": ["none", "distributed", "average"],
            "SEI on cracks": ["false", "true"],
            "SEI porosity change": ["false", "true"],
            "stress-induced diffusion": ["false", "true"],
            "surface form": ["false", "differential", "algebraic"],
            "thermal": ["isothermal", "lumped", "x-lumped", "x-full"],
            "total interfacial current density as a state": ["false", "true"],
            "working electrode": ["both", "negative", "positive"],
            "x-average side reactions": ["false", "true"],
        }

        default_options = {
            name: options[0] for name, options in self.possible_options.items()
        }

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

        # Change the default for particle mechanics based on which SEI on cracks and LAM
        # options are provided
        # return "false" and "none" respectively if options not given
        SEI_cracks_option = extra_options.get("SEI on cracks", "false")
        LAM_opt = extra_options.get("loss of active material", "none")
        if SEI_cracks_option == "true":
            if "stress-driven" in LAM_opt or "stress and reaction-driven" in LAM_opt:
                default_options["particle mechanics"] = (
                    "swelling and cracking",
                    "swelling only",
                )
            else:
                default_options["particle mechanics"] = (
                    "swelling and cracking",
                    "none",
                )
        else:
            if "stress-driven" in LAM_opt or "stress and reaction-driven" in LAM_opt:
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

        # Change the default for surface form based on which particle
        # phases option is provided.
        # return "1" if option not given
        phases_option = extra_options.get("particle phases", "1")
        if phases_option == "1":
            default_options["surface form"] = "false"
        else:
            default_options["surface form"] = "algebraic"
        # The "surface form" option will still be overridden by
        # extra_options if provided
        # Change default SEI model based on which lithium plating option is provided
        # return "none" if option not given
        plating_option = extra_options.get("lithium plating", "none")
        if plating_option == "partially reversible":
            default_options["SEI"] = "constant"
        else:
            default_options["SEI"] = "none"
        # The "SEI" option will still be overridden by extra_options if provided

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

        # If "SEI film resistance" is not "none" and there are multiple phases
        # then "total interfacial current density as a state" must be "true"
        if (
            options["SEI film resistance"] != "none"
            and options["particle phases"] != "1"
        ):
            options["total interfacial current density as a state"] = "true"
            # Check that extra_options did not try to provide a clashing option
            if (
                extra_options.get("total interfacial current density as a state")
                == "false"
            ):
                raise pybamm.OptionError(
                    "If 'SEI film resistance' is not 'none' "
                    "and there are multiple phases then 'total interfacial "
                    "current density as a state' must be 'true'"
                )

        # Options not yet compatible with contact resistance
        if options["contact resistance"] == "true":
            if options["operating mode"] == "explicit power":
                raise NotImplementedError(
                    "Contact resistance not yet supported for explicit power."
                )
            if options["operating mode"] == "explicit resistance":
                raise NotImplementedError(
                    "Contact resistance not yet supported for explicit resistance."
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

        if (
            options["thermal"] in ["x-lumped", "x-full"]
            and options["cell geometry"] != "pouch"
        ):
            raise pybamm.OptionError(
                options["thermal"] + " model must have pouch geometry."
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

        if options["working electrode"] != "both":
            if options["thermal"] == "x-full":
                raise pybamm.OptionError(
                    "X-full thermal submodel is not compatible with half-cell models"
                )
            elif options["thermal"] == "x-lumped" and options["dimensionality"] != 0:
                n = options["dimensionality"]
                raise pybamm.OptionError(
                    f"X-lumped thermal submodels do not yet support {n}D "
                    "current collectors in a half-cell configuration"
                )
            elif options["SEI on cracks"] == "true":
                raise NotImplementedError(
                    "SEI on cracks not yet implemented for half-cell models"
                )

        if options["particle phases"] != "1":
            if not (
                options["surface form"] != "false"
                and options["particle size"] == "single"
                and options["particle"] == "Fickian diffusion"
                and options["particle mechanics"] == "none"
                and options["loss of active material"] == "none"
                and options["lithium plating"] == "none"
            ):
                raise pybamm.OptionError(
                    "If there are multiple particle phases: 'surface form' cannot be "
                    "'false', 'particle size' must be 'single', 'particle' must be "
                    "'Fickian diffusion'. Also the following must "
                    "be 'none': 'particle mechanics', "
                    "'loss of active material', 'lithium plating'"
                )

        # Check options are valid
        for option, value in options.items():
            if option in ["working electrode"]:
                pass
            else:
                if isinstance(value, str) or option in [
                    "dimensionality",
                    "operating mode",
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
                                "open-circuit potential",
                                "particle",
                                "particle mechanics",
                                "particle phases",
                                "particle size",
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
                # flatten value
                value_list = []
                for val in value:
                    if isinstance(val, tuple):
                        value_list.extend(list(val))
                    else:
                        value_list.append(val)
                for val in value_list:
                    if val not in self.possible_options[option]:
                        if not (option == "operating mode" and callable(val)):
                            raise pybamm.OptionError(
                                f"\n'{val}' is not recognized in option '{option}'. "
                                f"Possible values are {self.possible_options[option]}"
                            )

        super().__init__(options.items())

    @property
    def phases(self):
        try:
            return self._phases
        except AttributeError:
            self._phases = {}
            for domain in ["negative", "positive"]:
                number = int(getattr(self, domain)["particle phases"])
                phases = ["primary"]
                if number >= 2:
                    phases.append("secondary")
                self._phases[domain] = phases
            return self._phases

    @cached_property
    def whole_cell_domains(self):
        if self["working electrode"] == "positive":
            return ["separator", "positive electrode"]
        elif self["working electrode"] == "negative":
            return ["negative electrode", "separator"]
        elif self["working electrode"] == "both":
            return ["negative electrode", "separator", "positive electrode"]

    @property
    def electrode_types(self):
        try:
            return self._electrode_types
        except AttributeError:
            self._electrode_types = {}
            for domain in ["negative", "positive"]:
                if f"{domain} electrode" in self.whole_cell_domains:
                    self._electrode_types[domain] = "porous"
                else:
                    self._electrode_types[domain] = "planar"
            return self._electrode_types

    def print_options(self):
        """
        Print the possible options with the ones currently selected
        """
        for key, value in self.items():
            print(f"{key!r}: {value!r} (possible: {self.possible_options[key]!r})")

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

    @property
    def primary(self):
        return BatteryModelPhaseOptions(self, 0)

    @property
    def secondary(self):
        return BatteryModelPhaseOptions(self, 1)


class BatteryModelPhaseOptions(dict):
    def __init__(self, domain_options, index):
        super().__init__(domain_options.items())
        self.domain_options = domain_options
        self.index = index

    def __getitem__(self, key):
        options = self.domain_options.__getitem__(key)
        if isinstance(options, str):
            return options
        else:
            # 2-tuple, first is primary phase, second is secondary phase
            return options[self.index]


class BaseBatteryModel(pybamm.BaseModel):
    """
    Base model class with some default settings and required variables

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
    """

    def __init__(self, options=None, name="Unnamed battery model"):
        super().__init__(name)
        self.options = options

    @property
    def default_geometry(self):
        return pybamm.battery_geometry(options=self.options)

    @property
    def default_var_pts(self):
        base_var_pts = {
            "x_n": 20,
            "x_s": 20,
            "x_p": 20,
            "r_n": 20,
            "r_p": 20,
            "r_n_prim": 20,
            "r_p_prim": 20,
            "r_n_sec": 20,
            "r_p_sec": 20,
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
            "negative primary particle": pybamm.Uniform1DSubMesh,
            "positive primary particle": pybamm.Uniform1DSubMesh,
            "negative secondary particle": pybamm.Uniform1DSubMesh,
            "positive secondary particle": pybamm.Uniform1DSubMesh,
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
            "negative primary particle": pybamm.FiniteVolume(),
            "positive primary particle": pybamm.FiniteVolume(),
            "negative secondary particle": pybamm.FiniteVolume(),
            "positive secondary particle": pybamm.FiniteVolume(),
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
        # if extra_options is a dict then process it into a BatteryModelOptions
        # this does not catch cases that subclass the dict type
        # so other submodels can pass in their own options class if needed
        if extra_options is None or type(extra_options) == dict:
            options = BatteryModelOptions(extra_options)
        else:
            options = extra_options

        # Options that are incompatible with models
        if isinstance(self, pybamm.lithium_ion.BaseModel):
            if options["convection"] != "none":
                raise pybamm.OptionError(
                    "convection not implemented for lithium-ion models"
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
        if isinstance(self, pybamm.lithium_ion.SPM) and not isinstance(
            self, pybamm.lithium_ion.SPMe
        ):
            if options["x-average side reactions"] == "false":
                raise pybamm.OptionError(
                    "x-average side reactions cannot be 'false' for SPM models"
                )
        if isinstance(self, pybamm.lithium_ion.SPM):
            if (
                "distribution" in options["particle size"]
                and options["surface form"] == "false"
            ):
                raise pybamm.OptionError(
                    "surface form must be 'algebraic' or 'differential' if "
                    " 'particle size' contains a 'distribution'"
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
            isinstance(self, pybamm.lead_acid.LOQS)
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
                "Time [s]": pybamm.t,
                "Time [min]": pybamm.t / 60,
                "Time [h]": pybamm.t / 3600,
            }
        )

        # Spatial
        var = pybamm.standard_spatial_vars
        self.variables.update(
            {"x [m]": var.x, "x_n [m]": var.x_n, "x_s [m]": var.x_s, "x_p [m]": var.x_p}
        )
        if self.options["dimensionality"] == 1:
            self.variables.update({"z [m]": var.z})
        elif self.options["dimensionality"] == 2:
            self.variables.update({"y [m]": var.y, "z [m]": var.z})

    def build_model_equations(self):
        # Set model equations
        for submodel_name, submodel in self.submodels.items():
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
        # Build model variables and equations
        self._build_model()

        # Set battery specific variables
        pybamm.logger.debug("Setting voltage variables ({})".format(self.name))
        self.set_voltage_variables()

        pybamm.logger.debug("Setting SoC variables ({})".format(self.name))
        self.set_soc_variables()

        pybamm.logger.debug("Setting degradation variables ({})".format(self.name))
        self.set_degradation_variables()
        self.set_summary_variables()

        self._built = True
        pybamm.logger.info("Finish building {}".format(self.name))

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
        options = getattr(self.options, domain)
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

    def get_inverse_intercalation_kinetics(self):
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
            model = pybamm.external_circuit.ExplicitCurrentControl(
                self.param, self.options
            )
        elif self.options["operating mode"] == "voltage":
            model = pybamm.external_circuit.VoltageFunctionControl(
                self.param, self.options
            )
        elif self.options["operating mode"] == "power":
            model = pybamm.external_circuit.PowerFunctionControl(
                self.param, self.options, "algebraic"
            )
        elif self.options["operating mode"] == "differential power":
            model = pybamm.external_circuit.PowerFunctionControl(
                self.param, self.options, "differential without max"
            )
        elif self.options["operating mode"] == "explicit power":
            model = pybamm.external_circuit.ExplicitPowerControl(
                self.param, self.options
            )
        elif self.options["operating mode"] == "resistance":
            model = pybamm.external_circuit.ResistanceFunctionControl(
                self.param, self.options, "algebraic"
            )
        elif self.options["operating mode"] == "differential resistance":
            model = pybamm.external_circuit.ResistanceFunctionControl(
                self.param, self.options, "differential without max"
            )
        elif self.options["operating mode"] == "explicit resistance":
            model = pybamm.external_circuit.ExplicitResistanceControl(
                self.param, self.options
            )
        elif self.options["operating mode"] == "CCCV":
            model = pybamm.external_circuit.CCCVFunctionControl(
                self.param, self.options
            )
        elif callable(self.options["operating mode"]):
            model = pybamm.external_circuit.FunctionControl(
                self.param, self.options["operating mode"], self.options
            )
        self.submodels["external circuit"] = model

    def set_transport_efficiency_submodels(self):
        self.submodels[
            "electrolyte transport efficiency"
        ] = pybamm.transport_efficiency.Bruggeman(
            self.param, "Electrolyte", self.options
        )
        self.submodels[
            "electrode transport efficiency"
        ] = pybamm.transport_efficiency.Bruggeman(self.param, "Electrode", self.options)

    def set_thermal_submodel(self):
        if self.options["thermal"] == "isothermal":
            thermal_submodel = pybamm.thermal.isothermal.Isothermal
        elif self.options["thermal"] == "lumped":
            thermal_submodel = pybamm.thermal.Lumped
        elif self.options["thermal"] == "x-lumped":
            if self.options["dimensionality"] == 0:
                thermal_submodel = pybamm.thermal.Lumped
            elif self.options["dimensionality"] == 1:
                thermal_submodel = pybamm.thermal.pouch_cell.CurrentCollector1D
            elif self.options["dimensionality"] == 2:
                thermal_submodel = pybamm.thermal.pouch_cell.CurrentCollector2D
        elif self.options["thermal"] == "x-full":
            if self.options["dimensionality"] == 0:
                thermal_submodel = pybamm.thermal.OneDimensionalX

        self.submodels["thermal"] = thermal_submodel(self.param, self.options)

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
        for domain in ["negative", "positive"]:
            Domain = domain.capitalize()
            util = getattr(self.options, domain)["interface utilisation"]
            if util == "full":
                submodel = pybamm.interface_utilisation.Full(
                    self.param, domain, self.options
                )
            elif util == "constant":
                submodel = pybamm.interface_utilisation.Constant(
                    self.param, domain, self.options
                )
            elif util == "current-driven":
                if self.options.electrode_types[domain] == "planar":
                    reaction_loc = "interface"
                elif self.x_average:
                    reaction_loc = "x-average"
                else:
                    reaction_loc = "full electrode"
                submodel = pybamm.interface_utilisation.CurrentDriven(
                    self.param, domain, self.options, reaction_loc
                )
            self.submodels[f"{Domain} interface utilisation"] = submodel

    def set_voltage_variables(self):
        if self.options.negative["particle phases"] == "1":
            # Only one phase, no need to distinguish between
            # "primary" and "secondary"
            phase_n = ""
        else:
            # add a space so that we can use "" or (e.g.) "primary " interchangeably
            # when naming variables
            phase_n = "primary "
        if self.options.positive["particle phases"] == "1":
            phase_p = ""
        else:
            phase_p = "primary "

        ocp_surf_n_av = self.variables[
            f"X-averaged negative electrode {phase_n}open-circuit potential [V]"
        ]
        ocp_surf_p_av = self.variables[
            f"X-averaged positive electrode {phase_p}open-circuit potential [V]"
        ]
        ocp_n_bulk = self.variables[
            f"Negative electrode {phase_n}bulk open-circuit potential [V]"
        ]
        ocp_p_bulk = self.variables[
            f"Positive electrode {phase_p}bulk open-circuit potential [V]"
        ]
        eta_particle_n = self.variables[
            f"Negative {phase_n}particle concentration overpotential [V]"
        ]
        eta_particle_p = self.variables[
            f"Positive {phase_p}particle concentration overpotential [V]"
        ]

        ocv_surf = ocp_surf_p_av - ocp_surf_n_av
        ocv_bulk = ocp_p_bulk - ocp_n_bulk

        eta_particle = eta_particle_p - eta_particle_n

        # overpotentials
        if self.options.electrode_types["negative"] == "planar":
            eta_r_n_av = self.variables[
                "Lithium metal interface reaction overpotential [V]"
            ]
        else:
            eta_r_n_av = self.variables[
                f"X-averaged negative electrode {phase_n}reaction overpotential [V]"
            ]
        eta_r_p_av = self.variables[
            f"X-averaged positive electrode {phase_p}reaction overpotential [V]"
        ]
        eta_r_av = eta_r_p_av - eta_r_n_av

        delta_phi_s_n_av = self.variables[
            "X-averaged negative electrode ohmic losses [V]"
        ]
        delta_phi_s_p_av = self.variables[
            "X-averaged positive electrode ohmic losses [V]"
        ]
        delta_phi_s_av = delta_phi_s_p_av - delta_phi_s_n_av

        # SEI film overpotential
        if self.options.electrode_types["negative"] == "planar":
            eta_sei_av = self.variables["SEI film overpotential [V]"]
        else:
            eta_sei_av = self.variables[
                f"X-averaged {phase_n}SEI film overpotential [V]"
            ]

        # TODO: add current collector losses to the voltage in 3D

        self.variables.update(
            {
                "Surface open-circuit voltage [V]": ocv_surf,
                "Open-circuit voltage [V]": ocv_bulk,
                "Particle concentration overpotential [V]": eta_particle,
                "X-averaged reaction overpotential [V]": eta_r_av,
                "X-averaged SEI film overpotential [V]": eta_sei_av,
                "X-averaged solid phase ohmic losses [V]": delta_phi_s_av,
            }
        )

        # Battery-wide variables
        V = self.variables["Voltage [V]"]
        eta_e_av = self.variables["X-averaged electrolyte ohmic losses [V]"]
        eta_c_av = self.variables["X-averaged concentration overpotential [V]"]
        num_cells = pybamm.Parameter(
            "Number of cells connected in series to make a battery"
        )
        self.variables.update(
            {
                "Battery open-circuit voltage [V]": ocv_bulk * num_cells,
                "Battery negative electrode bulk open-circuit potential [V]": ocp_n_bulk
                * num_cells,
                "Battery positive electrode bulk open-circuit potential [V]": ocp_p_bulk
                * num_cells,
                "Battery particle concentration overpotential [V]": eta_particle
                * num_cells,
                "Battery negative particle concentration overpotential [V]"
                "": eta_particle_n * num_cells,
                "Battery positive particle concentration overpotential [V]"
                "": eta_particle_p * num_cells,
                "X-averaged battery reaction overpotential [V]": eta_r_av * num_cells,
                "X-averaged battery negative reaction overpotential [V]": eta_r_n_av
                * num_cells,
                "X-averaged battery positive reaction overpotential [V]": eta_r_p_av
                * num_cells,
                "X-averaged battery solid phase ohmic losses [V]": delta_phi_s_av
                * num_cells,
                "X-averaged battery negative solid phase ohmic losses [V]"
                "": delta_phi_s_n_av * num_cells,
                "X-averaged battery positive solid phase ohmic losses [V]"
                "": delta_phi_s_p_av * num_cells,
                "X-averaged battery electrolyte ohmic losses [V]": eta_e_av * num_cells,
                "X-averaged battery concentration overpotential [V]": eta_c_av
                * num_cells,
                "Battery voltage [V]": V * num_cells,
            }
        )
        # Variables for calculating the equivalent circuit model (ECM) resistance
        # Need to compare OCV to initial value to capture this as an overpotential
        ocv_init = self.param.ocv_init
        eta_ocv = ocv_bulk - ocv_init
        # Current collector current density for working out euiqvalent resistance
        # based on Ohm's Law
        i_cc = self.variables["Current collector current density [A.m-2]"]
        # ECM overvoltage is OCV minus voltage
        v_ecm = ocv_bulk - V
        # Current collector area for turning resistivity into resistance
        A_cc = self.param.A_cc

        # Hack to avoid division by zero if i_cc is exactly zero
        # If i_cc is zero, i_cc_not_zero becomes 1. But multiplying by sign(i_cc) makes
        # the local resistance 'zero' (really, it's not defined when i_cc is zero)
        def x_not_zero(x):
            return ((x > 0) + (x < 0)) * x + (x >= 0) * (x <= 0)

        i_cc_not_zero = x_not_zero(i_cc)

        self.variables.update(
            {
                "Change in open-circuit voltage [V]": eta_ocv,
                "Local ECM resistance [Ohm]": pybamm.sign(i_cc)
                * v_ecm
                / (i_cc_not_zero * A_cc),
            }
        )

        # Cut-off voltage
        self.events.append(
            pybamm.Event(
                "Minimum voltage [V]",
                V - self.param.voltage_low_cut,
                pybamm.EventType.TERMINATION,
            )
        )
        self.events.append(
            pybamm.Event(
                "Maximum voltage [V]",
                self.param.voltage_high_cut - V,
                pybamm.EventType.TERMINATION,
            )
        )

        # Cut-off open-circuit voltage (for event switch with casadi 'fast with events'
        # mode)
        tol = 0.1
        self.events.append(
            pybamm.Event(
                "Minimum voltage switch [V]",
                V - (self.param.voltage_low_cut - tol),
                pybamm.EventType.SWITCH,
            )
        )
        self.events.append(
            pybamm.Event(
                "Maximum voltage switch [V]",
                V - (self.param.voltage_high_cut + tol),
                pybamm.EventType.SWITCH,
            )
        )

        # Power and resistance
        I = self.variables["Current [A]"]
        I_not_zero = x_not_zero(I)
        self.variables.update(
            {
                "Terminal power [W]": I * V,
                "Power [W]": I * V,
                "Resistance [Ohm]": pybamm.sign(I) * V / I_not_zero,
            }
        )

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
