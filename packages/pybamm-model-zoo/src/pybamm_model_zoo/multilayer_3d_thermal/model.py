"""A pouch cell stack resolved into zones, each with its own 3D temperature field."""

from __future__ import annotations

import functools
from collections.abc import Callable

import pybamm
import pybamm_model_zoo
from pybamm_model_zoo import _compat

SLUG = "multilayer_3d_thermal"
CONNECTIONS = ("parallel", "series")
#: How the electrodes are coated on their current collector foils.
COATINGS = ("double-sided", "single-sided")
#: The foil thickness parameters, each a whole foil.
FOIL_THICKNESSES = (
    "Negative current collector thickness [m]",
    "Positive current collector thickness [m]",
)
#: The stack's faces, as their heat transfer coefficient parameters name them.
FACES = ("Left", "Right", "Front", "Back", "Bottom", "Top")


#: The zone model's temperature state, heat source, and heat capacity, as
#: PyBaMM's lumped thermal submodel names them.
ZONE_TEMPERATURE = "Volume-averaged cell temperature [K]"
ZONE_HEATING = "Volume-averaged total heating [W.m-3]"
ZONE_HEAT_CAPACITY = "Volume-averaged effective heat capacity [J.K-1.m-3]"
#: Options the stack sets on every zone: a lumped temperature for the 3D field to
#: replace, no casing of its own, and a pouch unit cell.
ZONE_THERMAL_OPTIONS = {
    "thermal": "lumped",
    "surface temperature": "ambient",
    "cell geometry": "pouch",
    "dimensionality": 0,
}
#: The zone's heat terms in watts, summed over the stack under the same names.
HEAT_TERMS = (
    "Total heating [W]",
    "Ohmic heating [W]",
    "Irreversible electrochemical heating [W]",
    "Reversible heating [W]",
    "Heat of mixing [W]",
    "Hysteresis electrochemical heating [W]",
)
#: Zone variables kept under a ``"Layer i "`` prefix. A full model carries hundreds
#: of variables, and every one kept is processed for every zone.
ZONE_VARIABLES = {
    "Total current density [A.m-2]",
    "Battery open-circuit voltage [V]",
    "Surface open-circuit voltage [V]",
    "Discharge capacity [A.h]",
    "Throughput capacity [A.h]",
    "Total lithium in electrolyte [mol]",
    "Total lithium in particles [mol]",
    "Electrolyte concentration [mol.m-3]",
    "Electrolyte potential [V]",
    "Negative electrode potential [V]",
    "Positive electrode potential [V]",
    *HEAT_TERMS,
}
ZONE_VARIABLE_PREFIXES = ("X-averaged ", "Volume-averaged ")
ZONE_VARIABLE_FRAGMENTS = (
    "interfacial current density",
    "concentration",
    "stoichiometry",
    "open-circuit potential",
    "hysteresis state",
    "overpotential",
    "ohmic losses",
    "surface potential difference",
    "heating",
    "heat of mixing",
)


def _keep_zone_variable(name: str) -> bool:
    if name in ZONE_VARIABLES:
        return True
    return name.startswith(ZONE_VARIABLE_PREFIXES) and any(
        fragment in name for fragment in ZONE_VARIABLE_FRAGMENTS
    )


def _renamed(variable: pybamm.Variable, name: str) -> pybamm.Variable:
    return pybamm.Variable(
        name,
        domains=variable.domains,
        bounds=variable.bounds,
        scale=variable.scale,
        reference=variable.reference,
    )


def _state_renames(model: pybamm.BaseModel, prefix: str, names: dict) -> dict:
    """``{state: renamed state}`` for every state ``model`` owns.

    A state is renamed ``prefix + name`` unless ``names`` gives it a name of its
    own. A concatenated state (the electrolyte across the three regions) appears
    in the equations through its children, so each child is renamed and the
    concatenation is rebuilt from them.
    """
    mapping: dict = {}

    def rename(variable):
        if variable not in mapping:
            mapping[variable] = _renamed(
                variable, names.get(variable.name, prefix + variable.name)
            )
        return mapping[variable]

    for variable in [*model.rhs, *model.algebraic]:
        if isinstance(variable, pybamm.ConcatenationVariable):
            mapping[variable] = pybamm.concatenation(
                *(rename(child) for child in variable.children)
            )
        else:
            rename(variable)
    return mapping


class MultiLayer3DThermalSPM(pybamm.lithium_ion.BaseModel):
    """SPM zones through a pouch cell stack, each with its own 3D temperature.

    The ``num_physical_layers`` unit cells are lumped into ``num_subdivisions``
    zones. Each zone is PyBaMM's own :class:`pybamm.lithium_ion.SPM`, built with
    the model's options, for its unit cells in parallel; its lumped temperature
    is replaced by the volume average of a field ``T_i(x, y, z)`` on the domain
    ``"cell layer i"``, whose source is the zone's total heating. Adjacent zones
    exchange heat by conduction through the stack and a contact resistance;
    exposed faces are cooled convectively.

    Parameters
    ----------
    num_physical_layers : int, optional
        Unit cells in the stack, at least 2.
    num_subdivisions : int, optional
        Zones the stack is resolved into: at least 2, and a divisor of
        ``num_physical_layers``. Defaults to one zone per unit cell.
    connection : str, optional
        How the zones are connected, ``"parallel"`` or ``"series"``.
    mesh_h : float, optional
        Target element size of each zone's mesh, as for
        :class:`pybamm.ScikitFemGenerator3D`.
    options : dict, optional
        Model options, passed to every zone. ``"cell geometry"`` defaults to,
        and must be, ``"pouch"``; ``"thermal"`` must be ``"lumped"`` and
        ``"dimensionality"`` 0, since the stack's fields are the thermal model.
        ``"surface temperature"`` is recorded on the stack but not given to the
        zones, which have no casing of their own.
    name : str, optional
        The model name.
    zone_model : callable, optional
        ``zone_model(options)`` returns one zone's built model, in place of
        ``ZONE_MODEL(options=options)``. It is handed the zones' options and
        must honour them; use it to build a zone with a replaced submodel.
    coating : str, optional
        ``"double-sided"`` (default), each foil coated on both faces and shared
        by the two unit cells either side of it, so a unit cell carries half of
        each foil; or ``"single-sided"``, each unit cell with whole foils of its
        own. The current collector thicknesses are always whole foils.

    Raises
    ------
    pybamm.OptionError
        If the stack, connection, mesh size, or an option is invalid.
    pybamm.ModelError
        If a zone model lacks a lumped temperature, its heating, or its voltage.

    Examples
    --------
    >>> import pybamm_model_zoo as zoo
    >>> model = zoo.load("MultiLayer3DThermalSPM")(num_physical_layers=4)
    >>> "Temperature spread [K]" in model.variables
    True
    """

    #: Each zone's electrochemistry.
    ZONE_MODEL = pybamm.lithium_ion.SPM
    #: Thermal contact resistance between adjacent zones, on top of conduction
    #: through the zones themselves.
    CONTACT_RESISTANCE_PARAM = "Inter-layer thermal contact resistance [K.m2.W-1]"
    #: Perfect contact: the zones' own series conduction keeps the coupling well
    #: posed without it.
    DEFAULT_CONTACT_RESISTANCE = 0.0
    #: Supplied for any face heat transfer coefficient a parameter set lacks.
    DEFAULT_FACE_HEAT_TRANSFER_COEFFICIENT = 10.0
    #: The reference for each zone's electrochemistry.
    ELECTROCHEMISTRY_CITATION = "Marquis2019"

    def __init__(
        self,
        num_physical_layers: int = 3,
        num_subdivisions: int | None = None,
        connection: str = "parallel",
        mesh_h: float = 0.1,
        options: dict | None = None,
        name: str = "Multi-Layer 3D Thermal SPM",
        *,
        zone_model: Callable[[dict], pybamm.BaseModel] | None = None,
        coating: str = "double-sided",
    ) -> None:
        if num_subdivisions is None:
            num_subdivisions = num_physical_layers
        if num_physical_layers < 2:
            raise pybamm.OptionError(
                f"num_physical_layers must be at least 2, got {num_physical_layers}"
            )
        if num_subdivisions < 2:
            raise pybamm.OptionError(
                f"num_subdivisions must be at least 2, got {num_subdivisions}"
            )
        if num_physical_layers % num_subdivisions:
            raise pybamm.OptionError(
                "num_physical_layers must be divisible by num_subdivisions, got "
                f"{num_physical_layers} and {num_subdivisions}"
            )
        if connection not in CONNECTIONS:
            raise pybamm.OptionError(
                f"connection must be one of {list(CONNECTIONS)}, got '{connection}'"
            )
        if mesh_h <= 0:
            raise pybamm.OptionError(f"mesh_h must be positive, got {mesh_h}")
        if coating not in COATINGS:
            raise pybamm.OptionError(
                f"coating must be one of {list(COATINGS)}, got '{coating}'"
            )
        options = {"cell geometry": "pouch", "thermal": "lumped", **(options or {})}
        for key in ("cell geometry", "thermal", "dimensionality"):
            required = ZONE_THERMAL_OPTIONS[key]
            if options.get(key, required) != required:
                raise pybamm.OptionError(
                    f"{type(self).__name__} resolves its own temperature fields "
                    f"through a pouch cell, so '{key}' must be {required!r}, got "
                    f"{options[key]!r}"
                )

        super().__init__(options, name)
        pybamm_model_zoo.register_citation(SLUG)
        pybamm.citations.register(self.ELECTROCHEMISTRY_CITATION)

        self.num_physical_layers = num_physical_layers
        self.num_subdivisions = num_subdivisions
        self.layers_per_zone = num_physical_layers // num_subdivisions
        self.connection = connection
        self.mesh_h = mesh_h
        self.zone_options = {
            **{k: v for k, v in options.items() if k != "surface temperature"},
            **ZONE_THERMAL_OPTIONS,
        }
        self._zone_model = zone_model
        self.coating = coating
        # A shared foil is half each unit cell's: every symbol built from a foil
        # thickness, in the zones and the stack alike, sees its share.
        self._foil_shares = (
            {
                pybamm.Parameter(name): pybamm.Parameter(name) / 2
                for name in FOIL_THICKNESSES
            }
            if coating == "double-sided"
            else {}
        )

        self.thermal_variables = [
            pybamm.Variable(f"Layer {i} temperature [K]", domain=self._layer_domain(i))
            for i in range(num_subdivisions)
        ]
        self._layer_spatial_vars = {
            i: tuple(
                pybamm.SpatialVariable(axis, domain=self._layer_domain(i))
                for axis in "xyz"
            )
            for i in range(num_subdivisions)
        }
        self.layers = [
            self._build_electrochemistry_layer(i) for i in range(num_subdivisions)
        ]
        self._connect_layers()
        for i in range(num_subdivisions):
            self._set_layer_thermal(i)
        self._set_thermal_boundary_conditions()
        self._set_variables()
        self._set_voltage_events()

    @property
    def cells_in_parallel(self) -> int:
        """The unit cells that share the stack's current between them."""
        if self.connection == "parallel":
            return self.num_physical_layers
        return self.layers_per_zone

    def _layer_domain(self, layer_id: int) -> str:
        return f"cell layer {layer_id}"

    def _layer_current(
        self, layer_id: int
    ) -> tuple[pybamm.Variable | None, pybamm.Symbol]:
        """A zone's current unknown, if it has one, and its current.

        In parallel each zone's current is solved for, and so is free to differ
        from the others' and to flow at rest; in series every zone carries the
        stack's current. A zone's unit cells split its current equally.
        """
        if self.connection == "series":
            return None, self.param.current_with_time
        current = pybamm.Variable(f"Layer {layer_id} current [A]")
        return current, current

    def _build_zone(self) -> pybamm.BaseModel:
        options = dict(self.zone_options)
        if self._zone_model is not None:
            zone = self._zone_model(options)
        else:
            zone = self.ZONE_MODEL(options=options)
        missing = [
            name
            for name in ("Voltage [V]", ZONE_HEATING, ZONE_HEAT_CAPACITY)
            if name not in zone.variables
        ]
        if missing or not any(var.name == ZONE_TEMPERATURE for var in zone.rhs):
            raise pybamm.ModelError(
                f"A zone of {type(self).__name__} must be a built model with a "
                f"lumped temperature state '{ZONE_TEMPERATURE}'; "
                f"{type(zone).__name__} lacks {missing or [ZONE_TEMPERATURE]}"
            )
        return zone

    def _build_electrochemistry_layer(self, layer_id: int) -> dict:
        """One zone's model, renamed into the stack, and the symbols it couples by.

        Every state the zone owns is renamed ``"Layer i <name>"``, its applied
        current becomes one unit cell's share of the zone's, and its lumped
        temperature equation is dropped: the temperature becomes the algebraic
        volume average of the zone's field, set in ``_set_layer_thermal``.
        """
        zone = self._build_zone()
        prefix = f"Layer {layer_id} "
        unknown, current = self._layer_current(layer_id)
        temperature = next(var for var in zone.rhs if var.name == ZONE_TEMPERATURE)
        mapping = _state_renames(
            zone,
            prefix,
            {ZONE_TEMPERATURE: f"Layer {layer_id} average temperature [K]"},
        )
        mapping[zone.param.current_with_time] = current / self.layers_per_zone
        mapping.update(self._foil_shares)
        cache: dict = {}

        def substitute(symbol):
            return pybamm.replace(symbol, mapping, cache=cache)

        rhs = dict(zone.rhs)
        rhs.pop(temperature)
        initial_conditions = dict(zone.initial_conditions)
        initial_conditions.pop(temperature, None)
        self.rhs.update({substitute(k): substitute(v) for k, v in rhs.items()})
        self.algebraic.update(
            {substitute(k): substitute(v) for k, v in zone.algebraic.items()}
        )
        self.initial_conditions.update(
            {substitute(k): substitute(v) for k, v in initial_conditions.items()}
        )
        self.boundary_conditions.update(
            {
                substitute(var): {
                    side: (substitute(value), kind)
                    for side, (value, kind) in conditions.items()
                }
                for var, conditions in zone.boundary_conditions.items()
            }
        )
        self.events += [
            pybamm.Event(
                prefix + event.name, substitute(event.expression), event.event_type
            )
            for event in zone.events
        ]

        variables = {}
        for name, symbol in zone.variables.items():
            if not _keep_zone_variable(name):
                continue
            value = substitute(symbol)
            # A renamed state is registered under its own name, or not at all.
            if isinstance(value, pybamm.Variable) and value.name != prefix + name:
                continue
            variables[prefix + name] = value

        return {
            "T_av": mapping[temperature],
            "voltage": substitute(zone.variables["Voltage [V]"]),
            "current": current,
            "current_unknown": unknown,
            "heat": substitute(zone.variables[ZONE_HEATING]),
            "heat_capacity": substitute(zone.variables[ZONE_HEAT_CAPACITY]),
            "variables": variables,
        }

    def _connect_layers(self) -> None:
        voltages = [layer["voltage"] for layer in self.layers]
        if self.connection == "series":
            self._terminal_voltage = sum(voltages)
            return
        # Zone currents, not fractions of the stack's: a zone's voltage depends on
        # its own current, so this stays well posed at rest.
        stack_current = self.param.current_with_time
        currents = [layer["current_unknown"] for layer in self.layers]
        for current, voltage in zip(currents[1:], voltages[1:], strict=True):
            self.algebraic[current] = voltage - voltages[0]
        self.algebraic[currents[0]] = sum(currents) - stack_current
        for current in currents:
            self.initial_conditions[current] = stack_current / self.num_subdivisions
        self._terminal_voltage = voltages[0]

    def _set_layer_thermal(self, layer_id: int) -> None:
        domain = self._layer_domain(layer_id)
        T = self.thermal_variables[layer_id]
        layer = self.layers[layer_id]
        coordinates = list(self._layer_spatial_vars[layer_id])

        volume = pybamm.Integral(pybamm.PrimaryBroadcast(1.0, domain), coordinates)
        T_av = layer["T_av"]
        self.algebraic[T_av] = T_av - pybamm.Integral(T, coordinates) / volume
        self.initial_conditions[T_av] = self.param.T_init

        # The zone's heating and heat capacity are both per unit volume of its
        # unit cells, current collectors included, which is the zone's box.
        heat = _compat.source(pybamm.PrimaryBroadcast(layer["heat"], domain), T)
        conductivity = self.per_unit_cell(self.param.lambda_eff(T))
        self.rhs[T] = (
            conductivity * pybamm.laplacian(T)
            + pybamm.inner(pybamm.grad(conductivity), pybamm.grad(T))
            + heat
        ) / layer["heat_capacity"]
        self.initial_conditions[T] = pybamm.PrimaryBroadcast(self.param.T_init, domain)

    def zone_series_resistance(self, temperature: pybamm.Symbol) -> pybamm.Symbol:
        """Thermal resistance through one zone's thickness, layer by layer, in K.m2/W.

        ``lambda_eff`` is a thickness-weighted mean of the layers' conductivities,
        the in-plane value. Through the stack they conduct in series, which on a
        typical cell is one to two orders of magnitude less. Each zone's field
        carries ``lambda_eff``, so the series resistance between zone centres is
        put on the interfaces between them.
        """
        param = self.param
        unit_cell = (
            param.n.L_cc / param.n.lambda_cc(temperature)
            + param.n.L / param.n.lambda_(temperature)
            + param.s.L / param.s.lambda_(temperature)
            + param.p.L / param.p.lambda_(temperature)
            + param.p.L_cc / param.p.lambda_cc(temperature)
        )
        return self.per_unit_cell(self.layers_per_zone * unit_cell)

    def per_unit_cell(self, symbol: pybamm.Symbol) -> pybamm.Symbol:
        """``symbol`` with each foil thickness replaced by one unit cell's share."""
        if not self._foil_shares:
            return symbol
        return pybamm.replace(symbol, self._foil_shares)

    @property
    def unit_cell_thickness(self) -> pybamm.Symbol:
        """One unit cell's thickness: its electrodes, separator, and foil shares."""
        return self.per_unit_cell(self.param.L)

    def _set_thermal_boundary_conditions(self) -> None:
        """Equal and opposite Neumann fluxes between zones, since Dirichlet
        coupling of independent meshes over-constrains them; the zones' series
        conduction sits on these couplings, half a zone of it at each outer face."""
        param = self.param
        heat_transfer = {
            "x_min": param.h_edge_x_min,
            "x_max": param.h_edge_x_max,
            "y_min": param.h_edge_y_min,
            "y_max": param.h_edge_y_max,
            "z_min": param.h_edge_z_min,
            "z_max": param.h_edge_z_max,
        }
        contact_resistance = pybamm.Parameter(self.CONTACT_RESISTANCE_PARAM)
        neighbours = {"x_min": ("x_max", -1), "x_max": ("x_min", 1)}
        self._outer_face_temperatures = {}
        for i, T in enumerate(self.thermal_variables):
            _, y, z = self._layer_spatial_vars[i]
            T_amb = param.T_amb(y, z, pybamm.t)
            faces = {}
            for face, coefficient in heat_transfer.items():
                surface = pybamm.boundary_value(T, face)
                facing, step = neighbours.get(face, (None, 0))
                if step and 0 <= i + step < self.num_subdivisions:
                    j = i + step
                    neighbour = self.thermal_variables[j]
                    T_mean = (self.layers[i]["T_av"] + self.layers[j]["T_av"]) / 2
                    resistance = (
                        self.zone_series_resistance(T_mean) + contact_resistance
                    )
                    inflow = (
                        pybamm.boundary_value(neighbour, facing) - surface
                    ) / resistance
                elif step:
                    # An outer big face: half a zone of conduction, then the air.
                    half_zone = self.zone_series_resistance(self.layers[i]["T_av"]) / 2
                    inflow = (
                        coefficient * (T_amb - surface) / (1 + coefficient * half_zone)
                    )
                    T_amb_centre = param.T_amb(param.L_y / 2, param.L_z / 2, pybamm.t)
                    self._outer_face_temperatures[face] = (
                        surface + coefficient * half_zone * T_amb_centre
                    ) / (1 + coefficient * half_zone)
                else:
                    inflow = coefficient * (T_amb - surface)
                conductivity = pybamm.boundary_value(
                    self.per_unit_cell(param.lambda_eff(T)), face
                )
                faces[face] = (inflow / conductivity, "Neumann")
            self.boundary_conditions[T] = faces

    def _set_variables(self) -> None:
        current = self.param.current_with_time
        voltage = self._terminal_voltage
        num_cells = pybamm.Parameter(
            "Number of cells connected in series to make a battery"
        )
        self.variables.update(
            {
                "Current [A]": current,
                "Current variable [A]": current,
                # PyBaMM's per-unit-cell definition, over every unit cell in parallel.
                "Total current density [A.m-2]": current
                / (self.cells_in_parallel * self.param.A_cc),
                "Voltage [V]": voltage,
                "Terminal voltage [V]": voltage,
                "Battery voltage [V]": voltage * num_cells,
            }
        )

        n = self.layers_per_zone
        for i, layer in enumerate(self.layers):
            self.variables.update(layer["variables"])
            self.variables.update(
                {
                    f"Layer {i} temperature [K]": self.thermal_variables[i],
                    f"Layer {i} average temperature [K]": layer["T_av"],
                    f"Layer {i} heat generation [W.m-3]": layer["heat"],
                    f"Layer {i} heat capacity [J.K-1.m-3]": layer["heat_capacity"],
                    f"Layer {i} voltage [V]": layer["voltage"],
                    f"Layer {i} current [A]": layer["current"],
                    f"Layer {i} per-unit-cell current [A]": layer["current"] / n,
                }
            )
            if layer["current_unknown"] is not None:
                # Undefined at rest, where the stack's current is zero.
                self.variables[f"Layer {i} current fraction"] = (
                    layer["current"] / self.param.current_with_time
                )

        # Every zone has the same volume, so the stack average is their mean.
        averages = [layer["T_av"] for layer in self.layers]
        maximum = functools.reduce(pybamm.maximum, averages)
        minimum = functools.reduce(pybamm.minimum, averages)
        stack_average = sum(averages) / self.num_subdivisions
        # Each outer face is named as its heat transfer coefficient is, since
        # under one-sided cooling the two differ.
        left = self._outer_face_temperatures["x_min"]
        right = self._outer_face_temperatures["x_max"]
        surface = (left + right) / 2
        core = self._core_temperature()
        self.variables.update(
            {
                "Stack-averaged temperature [K]": stack_average,
                "Volume-averaged cell temperature [K]": stack_average,
                "X-averaged cell temperature [K]": stack_average,
                "Maximum layer-averaged temperature [K]": maximum,
                "Minimum layer-averaged temperature [K]": minimum,
                "Temperature spread [K]": maximum - minimum,
                "Left face temperature [K]": left,
                "Right face temperature [K]": right,
                "Surface temperature [K]": surface,
                "Core temperature [K]": core,
                "Core-to-skin temperature difference [K]": core - surface,
                "Volume-averaged total heating [W.m-3]": (
                    sum(layer["heat"] for layer in self.layers) / self.num_subdivisions
                ),
            }
        )

        # A zone's watts and amp-hours are one unit cell's, and it stands for n.
        # In series every zone passes the same charge, so the stack's is one zone's.
        for name in (
            *HEAT_TERMS,
            "Discharge capacity [A.h]",
            "Throughput capacity [A.h]",
        ):
            per_zone = [
                layer["variables"].get(f"Layer {i} {name}")
                for i, layer in enumerate(self.layers)
            ]
            if any(value is None for value in per_zone):
                continue
            if self.connection == "series" and name.endswith("[A.h]"):
                self.variables[name] = n * per_zone[0]
            else:
                self.variables[name] = n * sum(per_zone)

    def _core_temperature(self) -> pybamm.Symbol:
        """The stack's mid-plane temperature, averaged over the footprint.

        With an even number of zones the mid-plane is the interface between the
        middle two, whose faces differ by the drop across it, so it is their
        mean. With an odd number it falls inside the middle zone, which conducts
        with the in-plane ``lambda_eff`` and so is close to uniform through its
        thickness: its average.
        """
        middle = self.num_subdivisions // 2
        if self.num_subdivisions % 2:
            return self.layers[middle]["T_av"]
        return (
            pybamm.boundary_value(self.thermal_variables[middle - 1], "x_max")
            + pybamm.boundary_value(self.thermal_variables[middle], "x_min")
        ) / 2

    def _set_voltage_events(self) -> None:
        voltage = self._terminal_voltage
        # Zones in series each have to stay within the unit cell's limits.
        scale = self.num_subdivisions if self.connection == "series" else 1
        self.events += [
            pybamm.Event(
                "Minimum voltage [V]", voltage - scale * self.param.voltage_low_cut
            ),
            pybamm.Event(
                "Maximum voltage [V]", scale * self.param.voltage_high_cut - voltage
            ),
        ]

    def _with_model_defaults(
        self, parameter_values: pybamm.ParameterValues
    ) -> pybamm.ParameterValues:
        """Add this model's own parameters, where ``parameter_values`` lacks them."""
        defaults = {
            self.CONTACT_RESISTANCE_PARAM: self.DEFAULT_CONTACT_RESISTANCE,
            **{
                f"{face} face heat transfer coefficient [W.m-2.K-1]": (
                    self.DEFAULT_FACE_HEAT_TRANSFER_COEFFICIENT
                )
                for face in FACES
            },
        }
        present = set(parameter_values.keys())
        parameter_values.update(
            {key: value for key, value in defaults.items() if key not in present},
            check_already_exists=False,
        )
        return parameter_values

    def apply_stack_scaling(
        self, parameter_values: pybamm.ParameterValues
    ) -> pybamm.ParameterValues:
        """Scale a unit cell's parameter set up to the whole stack, in place.

        Multiplies ``"Nominal cell capacity [A.h]"`` by :attr:`cells_in_parallel`,
        so that a C-rate refers to the stack, and adds this model's own
        parameters where ``parameter_values`` lacks them.

        Parameters
        ----------
        parameter_values : pybamm.ParameterValues
            A unit cell's parameters. Modified in place.

        Returns
        -------
        pybamm.ParameterValues
            The same ``parameter_values``.
        """
        capacity = parameter_values["Nominal cell capacity [A.h]"]
        parameter_values["Nominal cell capacity [A.h]"] = (
            capacity * self.cells_in_parallel
        )
        pybamm.logger.info(
            f"{self.name}: {self.num_subdivisions} zones of {self.layers_per_zone} "
            f"unit cells, nominal capacity {capacity:g} A.h scaled by "
            f"{self.cells_in_parallel}"
        )
        return self._with_model_defaults(parameter_values)

    @property
    def default_parameter_values(self) -> pybamm.ParameterValues:
        return self._with_model_defaults(super().default_parameter_values)

    @property
    def default_geometry(self) -> pybamm.Geometry:
        geometry = pybamm.battery_geometry(options=self.options)
        # A zone spans its unit cells, current collectors included, so the zones
        # together span the stack and each box is the volume its heat is per.
        thickness = self.layers_per_zone * self.unit_cell_thickness
        for i in range(self.num_subdivisions):
            geometry[self._layer_domain(i)] = {
                "x": {"min": i * thickness, "max": (i + 1) * thickness},
                "y": {"min": 0, "max": self.param.L_y},
                "z": {"min": 0, "max": self.param.L_z},
            }
        return geometry

    @property
    def default_submesh_types(self) -> dict:
        submeshes = super().default_submesh_types
        for i in range(self.num_subdivisions):
            submeshes[self._layer_domain(i)] = pybamm.ScikitFemGenerator3D(
                geom_type="pouch", h=self.mesh_h
            )
        return submeshes

    @property
    def default_spatial_methods(self) -> dict:
        methods = super().default_spatial_methods
        for i in range(self.num_subdivisions):
            methods[self._layer_domain(i)] = pybamm.ScikitFiniteElement3D()
        return methods

    @property
    def default_var_pts(self) -> dict:
        # The 3D generator meshes from the geometry and `mesh_h`, not from var_pts.
        return {**super().default_var_pts, "x": None, "y": None, "z": None}
