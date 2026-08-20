"""Single Particle Model with a linearised open-circuit potential."""

from __future__ import annotations

import pybamm
import pybamm_model_zoo
from pybamm_model_zoo import _compat

SLUG = "linearised_spm"


class LinearisedOpenCircuitPotential(
    pybamm.open_circuit_potential.BaseOpenCircuitPotential
):
    """Open-circuit potential linearised about the initial stoichiometry.

    Replaces ``U(x)`` with its tangent at the stoichiometry the simulation starts
    from, ``U(x_0) + U'(x_0) (x - x_0)``. The gradient is PyBaMM's own symbolic
    derivative of whatever ``U`` the parameter set supplies, so no separate
    slope parameter is needed.
    """

    def get_coupled_variables(self, variables: dict) -> dict:
        _, Domain = self.domain_Domain
        phase_name = self.phase_name
        sto_surf, sto_bulk, T, T_bulk = self._get_stoichiometry_and_temperature(
            variables
        )
        sto_0 = self.phase_param.sto_init_av
        gradient = self.phase_param.U(sto_0, T).diff(sto_0)
        gradient_bulk = self.phase_param.U(sto_0, T_bulk).diff(sto_0)

        ocp_surf = self.phase_param.U(sto_0, T) + gradient * (sto_surf - sto_0)
        ocp_bulk = self.phase_param.U(sto_0, T_bulk) + gradient_bulk * (
            sto_bulk - sto_0
        )
        # Evaluated at the linearisation point too, so the model stays linear
        # in the state when a thermal submodel reads it.
        dUdT = self.phase_param.dUdT(sto_0)

        variables.update(self._get_standard_ocp_variables(ocp_surf, ocp_bulk, dUdT))
        self._alias_ocp_as_equilibrium(variables)
        variables.update(
            {
                f"{Domain} electrode {phase_name}linearisation stoichiometry": sto_0,
                f"{Domain} electrode {phase_name}linearisation open-circuit "
                "potential [V]": self.phase_param.U(sto_0, T_bulk),
                f"{Domain} electrode {phase_name}open-circuit potential "
                "gradient [V]": gradient_bulk,
            }
        )
        return variables


class LinearisedSPM(pybamm.lithium_ion.SPM):
    """SPM linearised about the stoichiometry it starts from, for GITT analysis.

    Both porous electrodes' open-circuit potentials are replaced by their
    tangent at the initial stoichiometry, and ``"intercalation kinetics"``
    defaults to ``"linear"``. Over a pulse short enough to stay near that point
    the voltage transient is the Weppner-Huggins ``sqrt(t)`` form, with a slope
    set by the diffusivity and by ``dU/dx``, which the model reports. Prefer
    :class:`pybamm.lithium_ion.SPM` for anything ranging far from the starting
    stoichiometry, where the tangent is no longer a good approximation.

    Parameters
    ----------
    options : dict, optional
        Model options, as for :class:`pybamm.lithium_ion.SPM`. The
        ``"open-circuit potential"`` option must be ``"single"``, since this
        model supplies its own.
    name : str, optional
        The model name.
    build : bool, optional
        Whether to build the model on instantiation.

    Examples
    --------
    >>> import pybamm_model_zoo as zoo
    >>> model = zoo.load("LinearisedSPM")()
    >>> "Positive electrode open-circuit potential gradient [V]" in model.variables
    True
    """

    def __init__(
        self,
        options: dict | None = None,
        name: str = "Linearised Single Particle Model",
        build: bool = True,
    ) -> None:
        super().__init__(
            options=_compat.spm_default_options(
                {"intercalation kinetics": "linear", **(options or {})}
            ),
            name=name,
            build=build,
        )
        pybamm_model_zoo.register_citation(
            SLUG, "PyBaMMModelZoo2026", "WeppnerHuggins1977"
        )

    def set_open_circuit_potential_submodel(self) -> None:
        # Let the base class wire up every electrode, then take over the porous
        # ones; a planar electrode's plating potential is not linearised.
        super().set_open_circuit_potential_submodel()
        for domain in ("negative", "positive"):
            if self.options.electrode_types[domain] != "porous":
                continue
            domain_options = getattr(self.options, domain)
            for phase in self.options.phases[domain]:
                option = getattr(domain_options, phase)["open-circuit potential"]
                if option != "single":
                    raise pybamm.OptionError(
                        f"LinearisedSPM supplies its own open-circuit potential, "
                        f"so it cannot also apply 'open-circuit potential': "
                        f"'{option}' in the {domain} electrode."
                    )
                self.submodels[f"{domain} {phase} open-circuit potential"] = (
                    LinearisedOpenCircuitPotential(
                        self.param,
                        domain,
                        "lithium-ion main",
                        self.options,
                        phase,
                        self.x_average,
                    )
                )
