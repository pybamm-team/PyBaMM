"""A pouch cell stack resolved into zones, each with its own 3D temperature field."""

from __future__ import annotations

import functools

import pybamm
import pybamm_model_zoo
from pybamm_model_zoo import _compat

SLUG = "multilayer_3d_thermal"
CONNECTIONS = ("parallel", "series")
#: The stack's faces, as their heat transfer coefficient parameters name them.
FACES = ("Left", "Right", "Front", "Back", "Bottom", "Top")


def electrolyte_transport(
    param: pybamm.LithiumIonParameters,
) -> tuple[pybamm.Symbol, pybamm.Symbol]:
    """Porosity and electrolyte transport efficiency across one unit cell.

    Parameters
    ----------
    param : pybamm.LithiumIonParameters
        The model's parameters.

    Returns
    -------
    tuple of pybamm.Symbol
        The porosity and the Bruggeman transport efficiency, each concatenated
        over the negative electrode, separator, and positive electrode.
    """
    regions = (
        ("Negative electrode", "negative electrode", param.n.b_e),
        ("Separator", "separator", param.s.b_e),
        ("Positive electrode", "positive electrode", param.p.b_e),
    )
    porosities = [
        pybamm.PrimaryBroadcast(pybamm.Parameter(f"{name} porosity"), domain)
        for name, domain, _ in regions
    ]
    efficiencies = [
        porosity**bruggeman
        for porosity, (_, _, bruggeman) in zip(porosities, regions, strict=True)
    ]
    return pybamm.concatenation(*porosities), pybamm.concatenation(*efficiencies)


def electrolyte_lithium(
    param: pybamm.LithiumIonParameters,
    concentrations: tuple[pybamm.Symbol, pybamm.Symbol, pybamm.Symbol],
) -> pybamm.Symbol:
    """Lithium in one unit cell's electrolyte, in mol.

    Summed region by region, because ``pybamm.x_average`` of the porosity times
    the concatenated concentration treats the piecewise porosity as uniform.

    Parameters
    ----------
    param : pybamm.LithiumIonParameters
        The model's parameters.
    concentrations : tuple of pybamm.Symbol
        The electrolyte concentration in the negative electrode, separator, and
        positive electrode.
    """
    regions = (
        ("Negative electrode", param.n.L),
        ("Separator", param.s.L),
        ("Positive electrode", param.p.L),
    )
    per_area = sum(
        pybamm.Parameter(f"{name} porosity") * thickness * pybamm.x_average(c_e)
        for (name, thickness), c_e in zip(regions, concentrations, strict=True)
    )
    return per_area * param.A_cc


class MultiLayer3DThermalSPM(pybamm.lithium_ion.BaseModel):
    """SPM zones through a pouch cell stack, each with its own 3D temperature.

    The ``num_physical_layers`` unit cells are lumped into ``num_subdivisions``
    zones. Each zone is one SPM for its unit cells in parallel, with a
    temperature field ``T_i(x, y, z)`` on the domain ``"cell layer i"`` whose
    volume average its kinetics and transport see. Adjacent zones exchange heat
    through a contact resistance; exposed faces are cooled convectively.

    Parameters
    ----------
    num_physical_layers : int, optional
        Unit cells in the stack, at least 2.
    num_subdivisions : int, optional
        Zones the stack is resolved into: at least 2, and a divisor of
        ``num_physical_layers``. Defaults to one zone per unit cell.
    connection : str, optional
        How the zones are connected, ``"parallel"`` or ``"series"``. The unit
        cells within a zone are always in parallel.
    mesh_h : float, optional
        Target element size of each zone's mesh, as for
        :class:`pybamm.ScikitFemGenerator3D`.
    options : dict, optional
        Model options. ``"cell geometry"`` defaults to, and must be, ``"pouch"``.
    name : str, optional
        The model name.

    Raises
    ------
    pybamm.OptionError
        If the stack, connection, mesh size, or cell geometry is invalid.

    Examples
    --------
    >>> import pybamm_model_zoo as zoo
    >>> model = zoo.load("MultiLayer3DThermalSPM")(num_physical_layers=4)
    >>> "Temperature spread [K]" in model.variables
    True
    """

    #: The thermal contact resistance between adjacent zones.
    CONTACT_RESISTANCE_PARAM = "Inter-layer thermal contact resistance [K.m2.W-1]"
    #: Close to perfect contact, but large enough to keep the coupling well posed.
    DEFAULT_CONTACT_RESISTANCE = 1e-4
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
        options = {"cell geometry": "pouch", **(options or {})}
        if options["cell geometry"] != "pouch":
            raise pybamm.OptionError(
                f"{type(self).__name__} stacks its layers through a pouch cell, so "
                f"'cell geometry' must be 'pouch', got '{options['cell geometry']}'"
            )

        super().__init__(options, name)
        pybamm_model_zoo.register_citation(SLUG)
        pybamm.citations.register(self.ELECTROCHEMISTRY_CITATION)

        self.num_physical_layers = num_physical_layers
        self.num_subdivisions = num_subdivisions
        self.layers_per_zone = num_physical_layers // num_subdivisions
        self.connection = connection
        self.mesh_h = mesh_h

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
    ) -> tuple[pybamm.Variable | None, pybamm.Symbol, pybamm.Symbol]:
        """A zone's current fraction, its current, and one unit cell's current density.

        The fraction is ``None`` in series, where every zone carries the whole
        current. A zone's unit cells split its current equally.
        """
        current = self.param.current_with_time
        current_density = self.param.current_density_with_time / self.layers_per_zone
        if self.connection == "series":
            return None, current, current_density
        fraction = pybamm.Variable(f"Layer {layer_id} current fraction")
        return fraction, fraction * current, fraction * current_density

    def _set_particle_diffusion(
        self,
        concentration: pybamm.Variable,
        flux: pybamm.Symbol,
        temperature: pybamm.Symbol,
        phase_param,
        initial_concentration: pybamm.Symbol,
    ) -> None:
        """Fickian diffusion in a particle, driven by the interfacial current density."""
        diffusivity = phase_param.D(concentration, temperature)
        self.rhs[concentration] = -pybamm.div(-diffusivity * pybamm.grad(concentration))
        self.boundary_conditions[concentration] = {
            "left": (pybamm.Scalar(0), "Neumann"),
            "right": (-flux / (self.param.F * pybamm.surf(diffusivity)), "Neumann"),
        }
        self.initial_conditions[concentration] = initial_concentration

    def _add_stoichiometry_events(
        self, prefix: str, sto_surf_n: pybamm.Symbol, sto_surf_p: pybamm.Symbol
    ) -> None:
        for electrode, sto in (("negative", sto_surf_n), ("positive", sto_surf_p)):
            self.events += [
                pybamm.Event(
                    f"{prefix} minimum {electrode} particle surface stoichiometry",
                    pybamm.min(sto) - 0.01,
                ),
                pybamm.Event(
                    f"{prefix} maximum {electrode} particle surface stoichiometry",
                    (1 - 0.01) - pybamm.max(sto),
                ),
            ]

    def _electrolyte_and_ohmic_losses(
        self,
        prefix: str,
        current_density: pybamm.Symbol,
        temperature: pybamm.Symbol,
        sto_surf_n: pybamm.Symbol,
        sto_surf_p: pybamm.Symbol,
    ) -> tuple[pybamm.Symbol, pybamm.Symbol, pybamm.Symbol, dict]:
        """The electrolyte a zone's kinetics see, and the voltage its transport loses.

        The SPM holds the electrolyte at its initial concentration and resolves
        no ohmic drop, so this is the hook an electrolyte-resolving zone overrides.

        Returns
        -------
        tuple
            The x-averaged electrolyte concentration in the negative and positive
            electrodes, the voltage lost to transport (negative on discharge), and
            any output variables the transport adds.
        """
        c_e = self.param.c_e_init_av
        return c_e, c_e, pybamm.Scalar(0), {}

    def _build_electrochemistry_layer(self, layer_id: int) -> dict:
        """One zone's SPM, and the symbols the stack couples it through."""
        param = self.param
        prefix = f"Layer {layer_id}"
        c_s_n = pybamm.Variable(
            f"{prefix} X-averaged negative particle concentration [mol.m-3]",
            domain="negative particle",
        )
        c_s_p = pybamm.Variable(
            f"{prefix} X-averaged positive particle concentration [mol.m-3]",
            domain="positive particle",
        )
        # Tied to the volume average of the zone's own field in _set_layer_thermal.
        T = pybamm.Variable(f"{prefix} average temperature [K]")
        fraction, current, i_cell = self._layer_current(layer_id)

        a_n = 3 * param.n.prim.epsilon_s_av / param.n.prim.R_typ
        a_p = 3 * param.p.prim.epsilon_s_av / param.p.prim.R_typ
        j_n = i_cell / (param.n.L * a_n)
        j_p = -i_cell / (param.p.L * a_p)
        self._set_particle_diffusion(
            c_s_n, j_n, T, param.n.prim, pybamm.x_average(param.n.prim.c_init)
        )
        self._set_particle_diffusion(
            c_s_p, j_p, T, param.p.prim, pybamm.x_average(param.p.prim.c_init)
        )
        c_s_surf_n = pybamm.surf(c_s_n)
        c_s_surf_p = pybamm.surf(c_s_p)
        sto_surf_n = c_s_surf_n / param.n.prim.c_max
        sto_surf_p = c_s_surf_p / param.p.prim.c_max
        self._add_stoichiometry_events(prefix, sto_surf_n, sto_surf_p)

        c_e_n, c_e_p, transport_loss, transport_variables = (
            self._electrolyte_and_ohmic_losses(
                prefix, i_cell, T, sto_surf_n, sto_surf_p
            )
        )
        RT_F = param.R * T / param.F
        j0_n = param.n.prim.j0(c_e_n, c_s_surf_n, T)
        j0_p = param.p.prim.j0(c_e_p, c_s_surf_p, T)
        eta_n = (2 / param.n.prim.ne) * RT_F * pybamm.arcsinh(j_n / (2 * j0_n))
        eta_p = (2 / param.p.prim.ne) * RT_F * pybamm.arcsinh(j_p / (2 * j0_p))
        ocv = param.p.prim.U(sto_surf_p, T) - param.n.prim.U(sto_surf_n, T)
        voltage = ocv + eta_p - eta_n + transport_loss

        # Reaction and entropic heat in each electrode, plus the transport losses,
        # which are dissipated ohmically; all per unit volume of the unit cell.
        heat_n = a_n * j_n * (eta_n + T * param.n.prim.dUdT(sto_surf_n))
        heat_p = a_p * j_p * (eta_p + T * param.p.prim.dUdT(sto_surf_p))
        heat = (
            heat_n * param.n.L + heat_p * param.p.L - i_cell * transport_loss
        ) / param.L_x

        return {
            "T_av": T,
            "voltage": voltage,
            "current": current,
            "current_fraction": fraction,
            "heat": heat,
            "variables": {
                c_s_n.name: c_s_n,
                c_s_p.name: c_s_p,
                f"{prefix} negative particle surface stoichiometry": sto_surf_n,
                f"{prefix} positive particle surface stoichiometry": sto_surf_p,
                f"{prefix} surface open-circuit voltage [V]": ocv,
                **transport_variables,
            },
        }

    def _connect_layers(self) -> None:
        voltages = [layer["voltage"] for layer in self.layers]
        if self.connection == "series":
            self._terminal_voltage = sum(voltages)
            return
        # In parallel every zone holds the terminal voltage, and together they
        # carry the whole current.
        fractions = [layer["current_fraction"] for layer in self.layers]
        for fraction, voltage in zip(fractions[1:], voltages[1:], strict=True):
            self.algebraic[fraction] = voltage - voltages[0]
        self.algebraic[fractions[0]] = sum(fractions) - 1
        for fraction in fractions:
            self.initial_conditions[fraction] = pybamm.Scalar(1 / self.num_subdivisions)
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

        heat = _compat.source(pybamm.PrimaryBroadcast(layer["heat"], domain), T)
        conductivity = self.param.lambda_eff(T)
        self.rhs[T] = (
            conductivity * pybamm.laplacian(T)
            + pybamm.inner(pybamm.grad(conductivity), pybamm.grad(T))
            + heat
        ) / self.param.rho_c_p_eff(T)
        self.initial_conditions[T] = pybamm.PrimaryBroadcast(self.param.T_init, domain)

    def _set_thermal_boundary_conditions(self) -> None:
        """Convective cooling on exposed faces, contact resistance between zones.

        Coupling two zones by Dirichlet conditions on their independent meshes
        over-constrains the system, so the interface flux ``(T_i - T_j) / R_th``
        enters both sides as equal and opposite Neumann conditions.
        """
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
        for i, T in enumerate(self.thermal_variables):
            _, y, z = self._layer_spatial_vars[i]
            T_amb = param.T_amb(y, z, pybamm.t)
            faces = {}
            for face, coefficient in heat_transfer.items():
                surface = pybamm.boundary_value(T, face)
                facing, step = neighbours.get(face, (None, 0))
                if step and 0 <= i + step < self.num_subdivisions:
                    neighbour = self.thermal_variables[i + step]
                    inflow = (
                        pybamm.boundary_value(neighbour, facing) - surface
                    ) / contact_resistance
                else:
                    inflow = coefficient * (T_amb - surface)
                conductivity = pybamm.boundary_value(param.lambda_eff(T), face)
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
                "Voltage [V]": voltage,
                "Terminal voltage [V]": voltage,
                "Battery voltage [V]": voltage * num_cells,
            }
        )

        for i, layer in enumerate(self.layers):
            self.variables.update(layer["variables"])
            self.variables.update(
                {
                    f"Layer {i} temperature [K]": self.thermal_variables[i],
                    f"Layer {i} average temperature [K]": layer["T_av"],
                    f"Layer {i} heat generation [W.m-3]": layer["heat"],
                    f"Layer {i} voltage [V]": layer["voltage"],
                    f"Layer {i} current [A]": layer["current"],
                    f"Layer {i} per-unit-cell current [A]": layer["current"]
                    / self.layers_per_zone,
                }
            )
            if layer["current_fraction"] is not None:
                self.variables[f"Layer {i} current fraction"] = layer[
                    "current_fraction"
                ]

        # Every zone has the same volume, so the stack average is their mean.
        averages = [layer["T_av"] for layer in self.layers]
        maximum = functools.reduce(pybamm.maximum, averages)
        minimum = functools.reduce(pybamm.minimum, averages)
        stack_average = sum(averages) / self.num_subdivisions
        self.variables.update(
            {
                "Stack-averaged temperature [K]": stack_average,
                "Volume-averaged cell temperature [K]": stack_average,
                "Maximum layer-averaged temperature [K]": maximum,
                "Minimum layer-averaged temperature [K]": minimum,
                "Temperature spread [K]": maximum - minimum,
            }
        )

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
        # A zone spans its unit cells, so the zones together span the stack.
        thickness = self.layers_per_zone * self.param.L_x
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
