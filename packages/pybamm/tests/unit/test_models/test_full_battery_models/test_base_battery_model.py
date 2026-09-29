#
# Tests for the base battery model class
#

import io
import os
import warnings
from contextlib import redirect_stdout

import pytest

import pybamm
from pybamm.models.full_battery_models import base_battery_model
from pybamm.models.full_battery_models.base_battery_model import (
    BatteryModelDomainOptions,
    BatteryModelOptions,
    active_electrodes,
    dependency_error,
    iter_option_leaves,
    join_electrode_values,
    option_values_match,
    replace_option_leaf,
    resolve_option,
    validate_option_value,
)
from pybamm.models.full_battery_models.lithium_metal.dfn import DFN as LithiumMetalDFN

OPTIONS_DICT = {
    "surface form": "differential",
    "loss of active material": "stress-driven",
    "thermal": "x-full",
    "cell geometry": "pouch",
    "particle mechanics": "swelling only",
    "stress-induced diffusion": "true",
}

PRINT_OPTIONS_OUTPUT = """\
'calculate discharge energy': 'false' (possible: ['false', 'true'])
'calculate heat source for isothermal models': 'false' (possible: ['false', 'true'])
'cell geometry': 'pouch' (possible: ['arbitrary', 'pouch', 'cylindrical'])
'contact resistance': 'false' (possible: ['false', 'true'])
'convection': 'none' (possible: ['none', 'uniform transverse', 'full transverse'])
'current collector': 'uniform' (possible: ['uniform', 'potential pair', 'potential pair quite conductive'])
'diffusivity': 'single' (possible: ['single', 'current sigmoid'])
'dimensionality': 0 (possible: [0, 1, 2, 3])
'electrolyte conductivity': 'default' (possible: ['default', 'full', 'leading order', 'composite', 'integrated'])
'exchange-current density': 'single' (possible: ['single', 'current sigmoid'])
'heat of mixing': 'false' (possible: ['false', 'true'])
'hydrolysis': 'false' (possible: ['false', 'true'])
'intercalation kinetics': 'symmetric Butler-Volmer' (possible: ['symmetric Butler-Volmer', 'asymmetric Butler-Volmer', 'linear', 'Marcus', 'Marcus-Hush-Chidsey', 'MSMR'])
'interface utilisation': 'full' (possible: ['full', 'constant', 'current-driven'])
'lithium plating': 'none' (possible: ['none', 'reversible', 'partially reversible', 'irreversible'])
'lithium plating porosity change': 'false' (possible: ['false', 'true'])
'loss of active material': 'stress-driven' (possible: ['none', 'stress-driven', 'asymmetric stress-driven', 'reaction-driven', 'current-driven', 'stress and reaction-driven', 'asymmetric stress and reaction-driven'])
'number of MSMR reactions': 'none' (possible: ['none'])
'open-circuit potential': 'single' (possible: ['single', 'current sigmoid', 'MSMR', 'one-state hysteresis', 'one-state differential capacity hysteresis'])
'operating mode': 'current' (possible: ['current', 'voltage', 'power', 'differential power', 'explicit power', 'resistance', 'differential resistance', 'explicit resistance', 'CCCV'])
'particle': 'Fickian diffusion' (possible: ['Fickian diffusion', 'uniform profile', 'quadratic profile', 'quartic profile', 'MSMR'])
'particle mechanics': 'swelling only' (possible: ['none', 'swelling only', 'swelling and cracking'])
'particle phases': '1' (possible: ['1', '2'])
'particle shape': 'spherical' (possible: ['spherical', 'no particles'])
'particle size': 'single' (possible: ['single', 'distribution'])
'SEI': 'none' (possible: ['none', 'constant', 'reaction limited', 'reaction limited (asymmetric)', 'solvent-diffusion limited', 'electron-migration limited', 'interstitial-diffusion limited', 'ec reaction limited', 'ec reaction limited (asymmetric)', 'VonKolzenberg2020', 'tunnelling limited'])
'SEI film resistance': 'none' (possible: ['none', 'distributed', 'average'])
'SEI on cracks': 'false' (possible: ['false', 'true'])
'SEI porosity change': 'false' (possible: ['false', 'true'])
'stress-induced diffusion': 'true' (possible: ['false', 'true'])
'surface form': 'differential' (possible: ['false', 'differential', 'algebraic'])
'surface temperature': 'ambient' (possible: ['ambient', 'lumped'])
'thermal': 'x-full' (possible: ['isothermal', 'lumped', 'x-lumped', 'x-full'])
'total interfacial current density as a state': 'false' (possible: ['false', 'true'])
'transport efficiency': 'Bruggeman' (possible: ['Bruggeman', 'ordered packing', 'hyperbola of revolution', 'overlapping spheres', 'tortuosity factor', 'random overlapping cylinders', 'heterogeneous catalyst', 'cation-exchange membrane'])
'voltage as a state': 'false' (possible: ['false', 'true'])
'working electrode': 'both' (possible: ['both', 'positive'])
'x-average side reactions': 'false' (possible: ['false', 'true'])
'use lumped thermal capacity': 'false' (possible: ['false', 'true'])
"""


class TestBaseBatteryModel:
    def test_symbol_processor(self):
        model = pybamm.lithium_ion.SPM()
        # Set up geometry and parameters
        geometry = model.default_geometry
        parameter_values = model.default_parameter_values
        parameter_values.process_geometry(geometry)
        parameter_values.process_model(model)
        # Set up discretisation
        mesh = pybamm.Mesh(geometry, model.default_submesh_types, model.default_var_pts)
        disc = pybamm.Discretisation(mesh, model.default_spatial_methods)
        disc.process_model(model)

        # Process expression. We need to get the original model variables to use with `observe`
        c = (
            pybamm.Parameter("Negative electrode thickness [m]")
            * model.variables["X-averaged negative particle concentration [mol.m-3]"]
        )
        processed_c = model.process_symbol(c)
        assert isinstance(processed_c, pybamm.Multiplication)
        assert isinstance(processed_c.left, pybamm.Scalar)
        assert isinstance(processed_c.right, pybamm.StateVector)
        # Process flux manually and check result against flux computed in particle
        # submodel
        c_n = model.variables["X-averaged negative particle concentration [mol.m-3]"]
        T = pybamm.PrimaryBroadcast(
            model.variables["X-averaged negative electrode temperature [K]"],
            ["negative particle"],
        )
        D = model.param.n.prim.D(c_n, T)
        N = -D * pybamm.grad(c_n)

        flux_1 = model.process_symbol(N)
        flux_2 = model.variables["X-averaged negative particle flux [mol.m-2.s-1]"]
        param_flux_2 = parameter_values.process_symbol(flux_2)
        disc_flux_2 = disc.process_symbol(param_flux_2)
        assert flux_1 == disc_flux_2

    def test_summary_variables(self):
        model = pybamm.BaseBatteryModel()
        model.variables["var"] = pybamm.Scalar(1)
        model.summary_variables = ["var"]
        assert model.summary_variables == ["var"]
        with pytest.raises(KeyError, match=r"No cycling variable defined"):
            model.summary_variables = ["bad var"]

    def test_default_geometry(self):
        model = pybamm.BaseBatteryModel({"dimensionality": 0})
        assert model.default_geometry["current collector"]["z"]["position"] == 1
        model = pybamm.BaseBatteryModel({"dimensionality": 1, "cell geometry": "pouch"})
        assert model.default_geometry["current collector"]["z"]["min"] == 0
        model = pybamm.BaseBatteryModel({"dimensionality": 2, "cell geometry": "pouch"})
        assert model.default_geometry["current collector"]["y"]["min"] == 0

    def test_default_submesh_types(self):
        model = pybamm.BaseBatteryModel({"dimensionality": 0})
        assert issubclass(
            model.default_submesh_types["current collector"],
            pybamm.SubMesh0D,
        )
        model = pybamm.BaseBatteryModel({"dimensionality": 1, "cell geometry": "pouch"})
        assert issubclass(
            model.default_submesh_types["current collector"],
            pybamm.Uniform1DSubMesh,
        )
        model = pybamm.BaseBatteryModel({"dimensionality": 2, "cell geometry": "pouch"})
        assert issubclass(
            model.default_submesh_types["current collector"],
            pybamm.ScikitUniform2DSubMesh,
        )
        model = pybamm.BaseBatteryModel({"dimensionality": 3, "cell geometry": "pouch"})
        assert issubclass(
            model.default_submesh_types["current collector"],
            pybamm.SubMesh0D,
        )
        assert isinstance(
            model.default_submesh_types["cell"],
            pybamm.ScikitFemGenerator3D,
        )

    def test_default_var_pts(self):
        var_pts = {
            "x_n": 20,
            "x_s": 20,
            "x_p": 20,
            "r_n": 20,
            "r_n_prim": 20,
            "r_n_sec": 20,
            "r_p": 20,
            "r_p_prim": 20,
            "r_p_sec": 20,
            "y": 10,
            "z": 10,
            "R_n": 30,
            "R_p": 30,
            "R_n_prim": 30,
            "R_p_prim": 30,
            "R_n_sec": 30,
            "R_p_sec": 30,
        }
        model = pybamm.BaseBatteryModel({"dimensionality": 0})
        assert var_pts == model.default_var_pts

        var_pts.update({"x_n": 10, "x_s": 10, "x_p": 10})
        model = pybamm.BaseBatteryModel({"dimensionality": 2, "cell geometry": "pouch"})
        assert var_pts == model.default_var_pts

    def test_default_spatial_methods(self):
        model = pybamm.BaseBatteryModel({"dimensionality": 0})
        assert isinstance(
            model.default_spatial_methods["current collector"],
            pybamm.ZeroDimensionalSpatialMethod,
        )
        model = pybamm.BaseBatteryModel({"dimensionality": 1, "cell geometry": "pouch"})
        assert isinstance(
            model.default_spatial_methods["current collector"], pybamm.FiniteVolume
        )
        model = pybamm.BaseBatteryModel({"dimensionality": 2, "cell geometry": "pouch"})
        assert isinstance(
            model.default_spatial_methods["current collector"],
            pybamm.ScikitFiniteElement,
        )
        model = pybamm.BaseBatteryModel({"dimensionality": 3, "cell geometry": "pouch"})
        assert isinstance(
            model.default_spatial_methods["current collector"],
            pybamm.ZeroDimensionalSpatialMethod,
        )
        assert isinstance(
            model.default_spatial_methods["cell"],
            pybamm.ScikitFiniteElement3D,
        )

    # exercises several legacy dependent defaults directly
    def test_options(self, allow_legacy_defaults):
        with pytest.raises(pybamm.OptionError, match=r"Option"):
            pybamm.BaseBatteryModel({"bad option": "bad option"})
        with pytest.raises(
            pybamm.OptionError, match=r"is not recognized in option 'current collector'"
        ):
            pybamm.BaseBatteryModel({"current collector": "bad current collector"})
        with pytest.raises(pybamm.OptionError, match=r"thermal"):
            pybamm.BaseBatteryModel({"thermal": "bad thermal"})
        with pytest.raises(pybamm.OptionError, match=r"cell geometry"):
            pybamm.BaseBatteryModel({"cell geometry": "bad geometry"})
        with pytest.raises(pybamm.OptionError, match=r"dimensionality"):
            pybamm.BaseBatteryModel({"dimensionality": 5})
        with pytest.raises(pybamm.OptionError, match=r"current collector"):
            pybamm.BaseBatteryModel(
                {"dimensionality": 1, "current collector": "bad option"}
            )
        with pytest.raises(pybamm.OptionError, match=r"1D current collectors"):
            pybamm.BaseBatteryModel(
                {
                    "current collector": "potential pair",
                    "dimensionality": 1,
                    "thermal": "x-full",
                    "cell geometry": "pouch",
                }
            )
        with pytest.raises(pybamm.OptionError, match=r"2D current collectors"):
            pybamm.BaseBatteryModel(
                {
                    "current collector": "potential pair",
                    "dimensionality": 2,
                    "thermal": "x-full",
                    "cell geometry": "pouch",
                }
            )
        with pytest.raises(pybamm.OptionError, match=r"surface form"):
            pybamm.BaseBatteryModel({"surface form": "bad surface form"})
        with pytest.raises(pybamm.OptionError, match=r"convection"):
            pybamm.BaseBatteryModel({"convection": "bad convection"})
        with pytest.raises(
            pybamm.OptionError, match=r"cannot have transverse convection in 0D model"
        ):
            pybamm.BaseBatteryModel({"convection": "full transverse"})
        with pytest.raises(pybamm.OptionError, match=r"particle"):
            pybamm.BaseBatteryModel({"particle": "bad particle"})
        with pytest.raises(pybamm.OptionError, match=r"working electrode"):
            pybamm.BaseBatteryModel({"working electrode": "bad working electrode"})
        with pytest.raises(pybamm.OptionError, match=r"The 'negative' working"):
            pybamm.BaseBatteryModel({"working electrode": "negative"})
        with pytest.raises(pybamm.OptionError, match=r"particle shape"):
            pybamm.BaseBatteryModel({"particle shape": "bad particle shape"})
        with pytest.raises(pybamm.OptionError, match=r"operating mode"):
            pybamm.BaseBatteryModel({"operating mode": "bad operating mode"})
        with pytest.raises(pybamm.OptionError, match=r"electrolyte conductivity"):
            pybamm.BaseBatteryModel(
                {"electrolyte conductivity": "bad electrolyte conductivity"}
            )

        # SEI options
        with pytest.raises(pybamm.OptionError, match=r"SEI"):
            pybamm.BaseBatteryModel({"SEI": "bad sei"})
        with pytest.raises(pybamm.OptionError, match=r"SEI film resistance"):
            pybamm.BaseBatteryModel({"SEI film resistance": "bad SEI film resistance"})
        with pytest.raises(pybamm.OptionError, match=r"SEI porosity change"):
            pybamm.BaseBatteryModel({"SEI porosity change": "bad SEI porosity change"})
        # changing defaults based on other options
        model = pybamm.BaseBatteryModel()
        assert model.options["SEI film resistance"] == "none"
        model = pybamm.BaseBatteryModel({"SEI": "constant"})
        assert model.options["SEI film resistance"] == "distributed"
        assert model.options["total interfacial current density as a state"] == "true"
        model = pybamm.BaseBatteryModel(
            {"SEI film resistance": "average", "particle phases": "2"}
        )
        assert model.options["total interfacial current density as a state"] == "true"
        with pytest.raises(
            pybamm.OptionError,
            match=r"'total interfacial current density as a state' to be 'true'",
        ):
            pybamm.BaseBatteryModel(
                {
                    "SEI film resistance": "distributed",
                    "total interfacial current density as a state": "false",
                }
            )
        with pytest.raises(
            pybamm.OptionError,
            match=r"'total interfacial current density as a state' to be 'true'",
        ):
            pybamm.BaseBatteryModel(
                {
                    "SEI film resistance": "average",
                    "particle phases": "2",
                    "total interfacial current density as a state": "false",
                }
            )

        # loss of active material model
        with pytest.raises(pybamm.OptionError, match=r"loss of active material"):
            pybamm.BaseBatteryModel({"loss of active material": "bad LAM model"})
        with pytest.raises(pybamm.OptionError, match=r"loss of active material"):
            # can't have a 3-tuple
            pybamm.BaseBatteryModel(
                {
                    "loss of active material": (
                        "bad LAM model",
                        "bad LAM model",
                        "bad LAM model",
                    )
                }
            )

        # check default options change
        model = pybamm.BaseBatteryModel(
            {"loss of active material": "stress-driven", "SEI on cracks": "true"}
        )
        assert model.options["particle mechanics"] == (
            "swelling and cracking",
            "swelling only",
        )
        assert model.options["stress-induced diffusion"] == "true"
        model = pybamm.BaseBatteryModel(
            {
                "working electrode": "positive",
                "loss of active material": "stress-driven",
                "SEI on cracks": "true",
            }
        )
        assert model.options["particle mechanics"] == "swelling and cracking"
        assert model.options["stress-induced diffusion"] == "true"

        # crack model
        with pytest.raises(pybamm.OptionError, match=r"particle mechanics"):
            pybamm.BaseBatteryModel({"particle mechanics": "bad particle cracking"})
        with pytest.raises(pybamm.OptionError, match=r"particle cracking"):
            pybamm.BaseBatteryModel({"particle cracking": "bad particle cracking"})

        # SEI on cracks
        with pytest.raises(pybamm.OptionError, match=r"SEI on cracks"):
            pybamm.BaseBatteryModel({"SEI on cracks": "bad SEI on cracks"})
        with pytest.raises(
            pybamm.OptionError, match=r"'SEI on cracks' at negative is 'true'"
        ):
            pybamm.BaseBatteryModel(
                {"SEI on cracks": "true", "particle mechanics": "swelling only"}
            )

        # plating model
        with pytest.raises(pybamm.OptionError, match=r"lithium plating"):
            pybamm.BaseBatteryModel({"lithium plating": "bad plating"})
        with pytest.raises(
            pybamm.OptionError, match=r"lithium plating porosity change"
        ):
            pybamm.BaseBatteryModel(
                {
                    "lithium plating porosity change": "bad lithium "
                    "plating porosity change"
                }
            )
        with pytest.raises(pybamm.OptionError, match=r"distributions"):
            pybamm.BaseBatteryModel(
                {
                    "particle size": "distribution",
                    "lithium plating porosity change": "true",
                }
            )

        # contact resistance
        with pytest.raises(pybamm.OptionError, match=r"contact resistance"):
            pybamm.BaseBatteryModel({"contact resistance": "bad contact resistance"})
        with pytest.raises(NotImplementedError, match=r"Contact resistance not yet"):
            pybamm.BaseBatteryModel(
                {
                    "contact resistance": "true",
                    "operating mode": "explicit power",
                }
            )
        with pytest.raises(NotImplementedError, match=r"Contact resistance not yet"):
            pybamm.BaseBatteryModel(
                {
                    "contact resistance": "true",
                    "operating mode": "explicit resistance",
                }
            )

        # stress-induced diffusion
        with pytest.raises(
            pybamm.OptionError,
            match=r"'stress-induced diffusion' at negative is 'true'",
        ):
            pybamm.BaseBatteryModel({"stress-induced diffusion": "true"})

        # hydrolysis
        with pytest.raises(pybamm.OptionError, match=r"surface formulation"):
            pybamm.lead_acid.LOQS({"hydrolysis": "true", "surface form": "false"})

        # timescale
        with pytest.raises(pybamm.OptionError, match=r"timescale"):
            pybamm.BaseBatteryModel({"timescale": "bad timescale"})

        # thermal x-lumped
        with pytest.raises(pybamm.OptionError, match=r"x-lumped"):
            pybamm.lithium_ion.BaseModel(
                {"cell geometry": "arbitrary", "thermal": "x-lumped"}
            )

        # thermal half-cell
        with pytest.raises(pybamm.OptionError, match=r"X-full"):
            pybamm.BaseBatteryModel(
                {"thermal": "x-full", "working electrode": "positive"}
            )
        with pytest.raises(pybamm.OptionError, match=r"X-lumped"):
            pybamm.BaseBatteryModel(
                {
                    "dimensionality": 2,
                    "thermal": "x-lumped",
                    "working electrode": "positive",
                }
            )

        # thermal heat of mixing
        with pytest.raises(NotImplementedError, match=r"Heat of mixing"):
            pybamm.BaseBatteryModel(
                {
                    "heat of mixing": "true",
                    "particle size": "distribution",
                }
            )

        # surface thermal model
        with pytest.raises(pybamm.OptionError, match=r"surface temperature"):
            pybamm.BaseBatteryModel(
                {"surface temperature": "lumped", "thermal": "x-full"}
            )

        # phases
        with pytest.raises(pybamm.OptionError, match=r"multiple particle phases"):
            pybamm.BaseBatteryModel({"particle phases": "2", "surface form": "false"})

        # msmr
        with pytest.raises(pybamm.OptionError, match=r"MSMR"):
            pybamm.BaseBatteryModel({"open-circuit potential": "MSMR"})
        with pytest.raises(pybamm.OptionError, match=r"MSMR"):
            pybamm.BaseBatteryModel({"particle": "MSMR"})
        with pytest.raises(pybamm.OptionError, match=r"MSMR"):
            pybamm.BaseBatteryModel({"intercalation kinetics": "MSMR"})
        with pytest.raises(pybamm.OptionError, match=r"MSMR"):
            pybamm.BaseBatteryModel(
                {"open-circuit potential": "MSMR", "particle": "MSMR"}
            )
        with pytest.raises(pybamm.OptionError, match=r"MSMR"):
            pybamm.BaseBatteryModel(
                {"open-circuit potential": "MSMR", "intercalation kinetics": "MSMR"}
            )
        with pytest.raises(pybamm.OptionError, match=r"MSMR"):
            pybamm.BaseBatteryModel(
                {"particle": "MSMR", "intercalation kinetics": "MSMR"}
            )
        with pytest.raises(pybamm.OptionError, match=r"MSMR"):
            pybamm.BaseBatteryModel(
                {
                    "open-circuit potential": "MSMR",
                    "particle": "MSMR",
                    "intercalation kinetics": "MSMR",
                    "number of MSMR reactions": "1.5",
                }
            )
        # MSMR inside a per-electrode tuple must be rejected cleanly, not slip
        # through to crash later as int('none') in parameter construction.
        with pytest.raises(pybamm.OptionError, match=r"MSMR"):
            pybamm.BaseBatteryModel({"open-circuit potential": ("MSMR", "single")})
        with pytest.raises(pybamm.OptionError, match=r"MSMR"):
            pybamm.BaseBatteryModel({"particle": ("MSMR", "Fickian diffusion")})
        # MSMR with the default "none" reaction count must also be rejected cleanly.
        with pytest.raises(pybamm.OptionError, match=r"MSMR"):
            pybamm.BaseBatteryModel(
                {
                    "open-circuit potential": "MSMR",
                    "particle": "MSMR",
                    "intercalation kinetics": "MSMR",
                }
            )

    def test_msmr_mixed_electrode_accepted(self):
        # Verify non-MSMR electrode's "none" reaction count is legitimate
        options = {
            "open-circuit potential": ("MSMR", "single"),
            "particle": ("MSMR", "Fickian diffusion"),
            "intercalation kinetics": ("MSMR", "symmetric Butler-Volmer"),
            "number of MSMR reactions": ("3", "none"),
        }
        model = pybamm.BaseBatteryModel(options)
        assert model.options["number of MSMR reactions"] == ("3", "none")

    def test_msmr_half_cell_does_not_validate_counter_electrode(self):
        # On a half cell the negative electrode is lithium metal, not a porous
        # MSMR electrode, so a scalar "MSMR" option must not force a reaction
        # count on it.
        options = {
            "working electrode": "positive",
            "open-circuit potential": "MSMR",
            "particle": "MSMR",
            "intercalation kinetics": "MSMR",
            "number of MSMR reactions": ("none", "6"),
        }
        model = pybamm.BaseBatteryModel(options)
        assert model.options["number of MSMR reactions"] == ("none", "6")

    def test_msmr_half_cell_still_validates_working_electrode(self):
        # The working electrode is still validated: a "none" count
        # there must be rejected cleanly rather than slipping through.
        with pytest.raises(pybamm.OptionError, match=r"positive electrode"):
            pybamm.BaseBatteryModel(
                {
                    "working electrode": "positive",
                    "open-circuit potential": "MSMR",
                    "particle": "MSMR",
                    "intercalation kinetics": "MSMR",
                    "number of MSMR reactions": ("6", "none"),
                }
            )

    def test_build_twice(self):
        model = pybamm.lithium_ion.SPM()  # need to pick a model to set vars and build
        with pytest.raises(pybamm.ModelError, match=r"Model already built"):
            model.build_model()

    def test_get_coupled_variables(self):
        model = pybamm.lithium_ion.BaseModel()
        model.submodels["current collector"] = pybamm.current_collector.Uniform(
            model.param
        )
        with pytest.raises(pybamm.ModelError, match=r"Missing variable"):
            model.build_model()

    def test_default_solver(self):
        model = pybamm.BaseBatteryModel()
        assert isinstance(model.default_solver, pybamm.IDAKLUSolver)

        # check that default_solver gives you a new solver, not an internal object
        solver = model.default_solver
        solver = pybamm.BaseModel()
        assert isinstance(model.default_solver, pybamm.IDAKLUSolver)
        assert isinstance(solver, pybamm.BaseModel)

        # check that adding algebraic variables gives algebraic solver
        a = pybamm.Variable("a")
        model.algebraic = {a: a - 1}
        assert isinstance(model.default_solver, pybamm.NonlinearSolver)

    def test_option_type(self):
        # no entry gets default options
        model = pybamm.BaseBatteryModel()
        assert isinstance(model.options, pybamm.BatteryModelOptions)

        # dict options get converted to BatteryModelOptions
        model = pybamm.BaseBatteryModel({"thermal": "isothermal"})
        assert isinstance(model.options, pybamm.BatteryModelOptions)

        # special dict types are not changed
        options = pybamm.FuzzyDict({"thermal": "isothermal"})
        model = pybamm.BaseBatteryModel(options)
        assert model.options == options

    def test_save_load_model(self):
        try:
            model = pybamm.lithium_ion.SPM()
            geometry = model.default_geometry
            param = model.default_parameter_values
            param.process_model(model)
            param.process_geometry(geometry)
            mesh = pybamm.Mesh(
                geometry, model.default_submesh_types, model.default_var_pts
            )
            disc = pybamm.Discretisation(mesh, model.default_spatial_methods)
            disc.process_model(model)

            # save model
            model.save_model(
                filename="test_base_battery_model",
                mesh=mesh,
            )
        finally:
            os.remove("test_base_battery_model.json")

    def test_voltage_as_state(self):
        model = pybamm.lithium_ion.SPM({"voltage as a state": "true"})
        assert model.options["voltage as a state"] == "true"
        assert isinstance(model.variables["Voltage [V]"], pybamm.Variable)
        assert "Voltage [V]" in [v.name for v in model.algebraic]

        model = pybamm.lithium_ion.SPM(
            {"voltage as a state": "true", "operating mode": "voltage"}
        )
        assert model.options["voltage as a state"] == "true"
        assert isinstance(model.variables["Voltage [V]"], pybamm.Variable)
        assert "Voltage [V]" in [v.name for v in model.algebraic]

    def test_explicit_modes_default_voltage_as_state(self, allow_legacy_defaults):
        for mode in ["explicit power", "explicit resistance"]:
            with pytest.warns(pybamm.OptionDefaultDeprecationWarning):
                options = pybamm.BatteryModelOptions({"operating mode": mode})
            assert options["voltage as a state"] == "true"

    def test_explicit_modes_reject_voltage_as_state_false(self):
        for mode in ["explicit power", "explicit resistance"]:
            with pytest.raises(pybamm.OptionError, match="voltage as a state"):
                pybamm.BatteryModelOptions(
                    {"operating mode": mode, "voltage as a state": "false"}
                )

    def test_voltage_as_state_requires_voltage_expression(self):
        model = pybamm.BaseBatteryModel({"voltage as a state": "true"})
        model.variables["Voltage [V]"] = pybamm.Variable("Voltage [V]")
        with pytest.raises(pybamm.ModelError, match="Voltage expression"):
            model._constrain_voltage_to_expression()


@pytest.mark.usefixtures("allow_legacy_defaults")
class TestLegacyDependentDefaults:
    def test_shorthand_is_not_rewritten(self):
        options = BatteryModelOptions({"SEI": "constant"})
        assert options["SEI"] == "constant"
        assert options.negative["SEI"] == "constant"
        assert options.positive["SEI"] == "none"

    def test_sei_film_resistance_follows_sei_per_electrode(self):
        # 5807: ("none", "none") is no SEI anywhere
        options = BatteryModelOptions({"SEI": ("none", "none")})
        assert options["SEI film resistance"] == "none"
        assert options["total interfacial current density as a state"] == "false"
        options = BatteryModelOptions({"SEI": ("none", "constant")})
        assert options["SEI film resistance"] == "distributed"
        assert options["total interfacial current density as a state"] == "true"
        options = BatteryModelOptions(
            {"particle phases": ("2", "1"), "SEI": (("none", "constant"), "none")}
        )
        assert options["SEI film resistance"] == "distributed"

    def test_partial_plating_defaults_sei_per_electrode(self):
        options = BatteryModelOptions({"lithium plating": "partially reversible"})
        assert options.negative["SEI"] == "constant"
        assert options.positive["SEI"] == "none"
        # plating-derived SEI does not switch on film resistance
        assert options["SEI film resistance"] == "none"
        options = BatteryModelOptions(
            {"lithium plating": ("none", "partially reversible")}
        )
        assert options.negative["SEI"] == "none"
        assert options.positive["SEI"] == "constant"
        options = BatteryModelOptions(
            {"lithium plating": ("partially reversible", "partially reversible")}
        )
        assert options.negative["SEI"] == "constant"
        assert options.positive["SEI"] == "constant"
        options = BatteryModelOptions(
            {
                "particle phases": ("2", "1"),
                "lithium plating": (("partially reversible", "none"), "none"),
            }
        )
        assert options.negative["SEI"] == "constant"
        assert options.positive["SEI"] == "none"
        options = BatteryModelOptions(
            {"working electrode": "positive", "lithium plating": "partially reversible"}
        )
        assert options.positive["SEI"] == "constant"

    def test_mechanics_defaults_per_electrode(self):
        options = BatteryModelOptions({"SEI on cracks": "true"})
        assert options.negative["particle mechanics"] == "swelling and cracking"
        assert options.positive["particle mechanics"] == "none"
        options = BatteryModelOptions({"SEI on cracks": ("false", "true")})
        assert options.negative["particle mechanics"] == "none"
        assert options.positive["particle mechanics"] == "swelling and cracking"
        options = BatteryModelOptions(
            {
                "loss of active material": "stress-driven",
                "SEI on cracks": ("false", "true"),
            }
        )
        assert options.negative["particle mechanics"] == "swelling only"
        assert options.positive["particle mechanics"] == "swelling and cracking"
        options = BatteryModelOptions(
            {"loss of active material": ("stress-driven", "none")}
        )
        assert options.negative["particle mechanics"] == "swelling only"
        assert options.positive["particle mechanics"] == "none"

    def test_stress_diffusion_default_only_where_mechanics(self):
        # 4943
        options = BatteryModelOptions(
            {"particle mechanics": ("swelling and cracking", "none")}
        )
        assert options.negative["stress-induced diffusion"] == "true"
        assert options.positive["stress-induced diffusion"] == "false"
        options = BatteryModelOptions({"particle mechanics": "swelling only"})
        assert options["stress-induced diffusion"] == "true"
        options = BatteryModelOptions({})
        assert options["stress-induced diffusion"] == "false"
        options = BatteryModelOptions(
            {
                "particle phases": ("2", "1"),
                "particle mechanics": (("swelling only", "none"), "none"),
            }
        )
        assert options.negative.primary["stress-induced diffusion"] == "true"
        assert options.negative.secondary["stress-induced diffusion"] == "false"
        assert options.positive["stress-induced diffusion"] == "false"

    def test_all_single_phase_tuple_is_single_phase(self):
        # 3532 / 5680 / 4910
        options = BatteryModelOptions({"particle phases": ("1", "1")})
        assert options["surface form"] == "false"
        options = BatteryModelOptions(
            {
                "particle phases": ("1", "1"),
                "SEI": "constant",
                "SEI film resistance": "average",
            }
        )
        assert options["total interfacial current density as a state"] == "false"
        options = BatteryModelOptions({"particle phases": ("2", "1")})
        assert options["surface form"] == "algebraic"

    def test_supplied_values_are_never_overridden(self):
        options = BatteryModelOptions(
            {
                "SEI": "reaction limited",
                "SEI film resistance": "average",
                "particle mechanics": ("swelling only", "none"),
                "stress-induced diffusion": "false",
                "dimensionality": 1,
                "cell geometry": "arbitrary",
            }
        )
        assert options["SEI film resistance"] == "average"
        assert options["stress-induced diffusion"] == "false"
        assert options["cell geometry"] == "arbitrary"
        assert options.negative["particle mechanics"] == "swelling only"
        assert options.positive["particle mechanics"] == "none"

    def test_distributed_film_resistance_rejects_explicit_false_state(self):
        with pytest.raises(
            pybamm.OptionError,
            match=r"Option 'SEI film resistance' is 'distributed', which requires "
            r"'total interfacial current density as a state' to be 'true'",
        ):
            BatteryModelOptions(
                {
                    "SEI": "constant",
                    "SEI film resistance": "distributed",
                    "total interfacial current density as a state": "false",
                }
            )

    def test_multi_phase_film_resistance_rejects_explicit_false_state(self):
        with pytest.raises(
            pybamm.OptionError,
            match=r"'total interfacial current density as a state' to be 'true'",
        ):
            BatteryModelOptions(
                {
                    "particle phases": ("2", "1"),
                    "surface form": "algebraic",
                    "SEI": "constant",
                    "SEI film resistance": "average",
                    "total interfacial current density as a state": "false",
                }
            )

    def test_stress_driven_lam_requires_particle_mechanics(self):
        with pytest.raises(
            pybamm.OptionError,
            match=r"Option 'loss of active material' at negative is "
            r"'stress-driven', which requires 'particle mechanics' to be a "
            r"model other than 'none'",
        ):
            BatteryModelOptions(
                {
                    "loss of active material": "stress-driven",
                    "particle mechanics": "none",
                }
            )
        with pytest.raises(
            pybamm.OptionError,
            match=r"Option 'loss of active material' at positive is "
            r"'stress-driven', which requires 'particle mechanics' to be a "
            r"model other than 'none'",
        ):
            BatteryModelOptions(
                {
                    "loss of active material": ("none", "stress-driven"),
                    "particle mechanics": ("swelling only", "none"),
                }
            )
        options = BatteryModelOptions(
            {
                "loss of active material": ("stress-driven", "none"),
                "particle mechanics": ("swelling only", "none"),
            }
        )
        assert options.negative["particle mechanics"] == "swelling only"
        assert options.positive["particle mechanics"] == "none"


class TestElectrodeCompatibility:
    @pytest.mark.parametrize(
        "plating, sei, path",
        [
            ("partially reversible", "none", "negative"),
            (("none", "partially reversible"), "constant", "positive"),
            (
                ("partially reversible", "partially reversible"),
                ("constant", "none"),
                "positive",
            ),
        ],
    )
    def test_partial_plating_requires_sei(self, plating, sei, path):
        # 5709
        with pytest.raises(
            pybamm.OptionError,
            match=rf"Option 'lithium plating' at {path} is 'partially reversible', "
            r"which requires 'SEI' to be a model other than 'none'",
        ):
            BatteryModelOptions({"lithium plating": plating, "SEI": sei})

    def test_partial_plating_with_sei_is_valid(self):
        BatteryModelOptions(
            {
                "lithium plating": "partially reversible",
                "SEI": "constant",
                "SEI film resistance": "distributed",
                "total interfacial current density as a state": "true",
            }
        )
        BatteryModelOptions(
            {
                "lithium plating": ("none", "partially reversible"),
                "SEI": ("none", "constant"),
                "SEI film resistance": "distributed",
                "total interfacial current density as a state": "true",
            }
        )

    def test_partial_plating_sei_none_fails_at_model_construction(self):
        with pytest.raises(pybamm.OptionError, match=r"'lithium plating'"):
            pybamm.lithium_ion.DFN(
                {"lithium plating": "partially reversible", "SEI": "none"}
            )

    @pytest.mark.parametrize(
        "options",
        [
            {
                "lithium plating": "partially reversible",
                "SEI": "constant",
                "SEI film resistance": "none",
            },
            {
                "lithium plating": "partially reversible",
                "SEI": "constant",
                "SEI film resistance": "distributed",
                "total interfacial current density as a state": "true",
            },
        ],
    )
    def test_partial_plating_processes_okane2022(self, options):
        # 5709
        model = pybamm.lithium_ion.DFN(options)
        pybamm.ParameterValues("OKane2022").process_model(model)

    def test_partial_plating_half_cell(self):
        with pytest.raises(
            pybamm.OptionError,
            match=r"Option 'lithium plating' at positive is 'partially reversible'",
        ):
            BatteryModelOptions(
                {
                    "working electrode": "positive",
                    "lithium plating": "partially reversible",
                    "SEI": ("constant", "none"),
                }
            )
        # a scalar applies to the working (positive) electrode in a half cell
        BatteryModelOptions(
            {
                "working electrode": "positive",
                "lithium plating": "partially reversible",
                "SEI": "constant",
                "SEI film resistance": "distributed",
                "total interfacial current density as a state": "true",
            }
        )

    def test_sei_on_cracks_requires_cracking_per_electrode(self):
        with pytest.raises(
            pybamm.OptionError,
            match=r"Option 'SEI on cracks' at positive is 'true', which requires "
            r"'particle mechanics' to be 'swelling and cracking'",
        ):
            BatteryModelOptions(
                {
                    "SEI on cracks": ("true", "true"),
                    "particle mechanics": ("swelling and cracking", "swelling only"),
                }
            )

    def test_stress_diffusion_requires_mechanics_per_electrode(self):
        # 4943
        with pytest.raises(
            pybamm.OptionError,
            match=r"Option 'stress-induced diffusion' at positive is 'true', which "
            r"requires 'particle mechanics' to be a model other than 'none'",
        ):
            BatteryModelOptions(
                {
                    "particle mechanics": ("swelling only", "none"),
                    "stress-induced diffusion": "true",
                }
            )
        BatteryModelOptions(
            {
                "particle mechanics": ("swelling only", "none"),
                "stress-induced diffusion": ("true", "false"),
            }
        )

    def test_multi_phase_requirements_per_electrode(self):
        with pytest.raises(
            pybamm.OptionError,
            match=r"at negative has multiple particle phases",
        ):
            BatteryModelOptions(
                {"particle phases": ("2", "1"), "surface form": "false"}
            )
        with pytest.raises(pybamm.OptionError, match=r"'Fickian diffusion'"):
            BatteryModelOptions(
                {
                    "particle phases": ("2", "1"),
                    "particle": (
                        ("Fickian diffusion", "uniform profile"),
                        "Fickian diffusion",
                    ),
                }
            )
        # a non-Fickian particle in the single-phase electrode is fine
        BatteryModelOptions(
            {
                "particle phases": ("2", "1"),
                "particle": ("Fickian diffusion", "uniform profile"),
                "surface form": "algebraic",
            }
        )
        # particle-size distributions remain supported for composite electrodes
        BatteryModelOptions(
            {
                "particle phases": ("2", "1"),
                "particle size": "distribution",
                "surface form": "algebraic",
            }
        )


class TestOptions:
    def test_print_options(self):
        with io.StringIO() as buffer, redirect_stdout(buffer):
            BatteryModelOptions(OPTIONS_DICT).print_options()
            output = buffer.getvalue()

        assert output == PRINT_OPTIONS_OUTPUT

    def test_option_phases(self):
        options = BatteryModelOptions({})
        assert options.phases == {"negative": ["primary"], "positive": ["primary"]}
        options = BatteryModelOptions(
            {"particle phases": ("1", "2"), "surface form": "algebraic"}
        )
        assert options.phases == {
            "negative": ["primary"],
            "positive": ["primary", "secondary"],
        }

    def test_domain_options(self):
        options = BatteryModelOptions(
            {"particle": ("Fickian diffusion", "quadratic profile")}
        )
        assert options.negative["particle"] == "Fickian diffusion"
        assert options.positive["particle"] == "quadratic profile"
        # something that is the same in both domains
        assert options.negative["thermal"] == "isothermal"
        assert options.positive["thermal"] == "isothermal"

    def test_domain_phase_options(self):
        options = BatteryModelOptions(
            {
                "particle mechanics": (
                    ("swelling only", "swelling and cracking"),
                    "none",
                ),
                "stress-induced diffusion": ("true", "false"),
            }
        )
        assert options.negative["particle mechanics"] == (
            "swelling only",
            "swelling and cracking",
        )
        assert options.negative.primary["particle mechanics"] == "swelling only"
        assert (
            options.negative.secondary["particle mechanics"] == "swelling and cracking"
        )
        assert options.positive["particle mechanics"] == "none"
        assert options.positive.primary["particle mechanics"] == "none"
        assert options.positive.secondary["particle mechanics"] == "none"

    def test_whole_cell_domains(self):
        options = BatteryModelOptions({"working electrode": "positive"})
        assert options.whole_cell_domains == ["separator", "positive electrode"]

        options = BatteryModelOptions({})
        assert options.whole_cell_domains == [
            "negative electrode",
            "separator",
            "positive electrode",
        ]

    @pytest.mark.parametrize(
        "ocp_option",
        [
            ["Axen", "one-state hysteresis"],
            ["Wycisk", "one-state differential capacity hysteresis"],
            [("Axen", "single"), ("one-state hysteresis", "single")],
            [
                ("Wycisk", "single"),
                ("one-state differential capacity hysteresis", "single"),
            ],
            [
                ("Axen", "Wycisk"),
                ("one-state hysteresis", "one-state differential capacity hysteresis"),
            ],
            [
                (("Axen", "Wycisk"), "single"),
                (
                    (
                        "one-state hysteresis",
                        "one-state differential capacity hysteresis",
                    ),
                    "single",
                ),
            ],
        ],
    )
    def test_renamed_hysteresis_ocp(self, ocp_option):
        # check old option is renamed to new option
        assert (
            BatteryModelOptions({"open-circuit potential": ocp_option[0]}).get(
                "open-circuit potential"
            )
            == ocp_option[1]
        )

    def test_default_options_independent_of_possible_options_order(self):
        """Defaults should be set explicitly, not derived from possible_options[0]."""
        options = pybamm.BatteryModelOptions({})
        # defaults are set explicitly, not derived from list order
        assert options["voltage as a state"] == "false"
        # surface form: possible_options lists "false" first, default is "false"
        assert options["surface form"] == "false"

    def test_default_options_cover_all_possible_options(self):
        """Every key in possible_options must have an explicit default."""
        options = pybamm.BatteryModelOptions({})
        for key in options.possible_options:
            assert key in options, f"Missing default for option '{key}'"

    def test_input_not_mutated(self):
        supplied = {
            "SEI": "constant",
            "SEI on cracks": "true",
            "lithium plating": "reversible",
            "open-circuit potential": ("Axen", "single"),
            "SEI film resistance": "distributed",
            "particle mechanics": ("swelling and cracking", "none"),
            "stress-induced diffusion": ("true", "false"),
            "total interfacial current density as a state": "true",
        }
        snapshot = dict(supplied)
        BatteryModelOptions(supplied)
        assert supplied == snapshot

    def test_renamed_ocp_nested_three_levels(self):
        options = BatteryModelOptions(
            {
                "particle phases": ("2", "1"),
                "open-circuit potential": (("Axen", "Wycisk"), "Axen"),
                "surface form": "algebraic",
            }
        )
        assert options.negative.primary["open-circuit potential"] == (
            "one-state hysteresis"
        )
        assert options.negative.secondary["open-circuit potential"] == (
            "one-state differential capacity hysteresis"
        )
        assert options.positive["open-circuit potential"] == "one-state hysteresis"

    def test_rejects_phase_kinetics(self):
        with pytest.raises(pybamm.OptionError, match=r"Per-phase values"):
            BatteryModelOptions(
                {
                    "intercalation kinetics": (
                        ("symmetric Butler-Volmer", "asymmetric Butler-Volmer"),
                        "symmetric Butler-Volmer",
                    )
                }
            )

    def test_accepts_phase_exchange_current_density(self):
        options = BatteryModelOptions(
            {
                "exchange-current density": (
                    ("current sigmoid", "single"),
                    "single",
                )
            }
        )
        assert options.negative.primary["exchange-current density"] == "current sigmoid"
        assert options.negative.secondary["exchange-current density"] == "single"

    def test_invalid_leaf_reports_path(self):
        with pytest.raises(
            pybamm.OptionError,
            match=r"'bad' is not recognized in option 'particle' at positive",
        ):
            BatteryModelOptions({"particle": ("Fickian diffusion", "bad")})

    def test_invalid_shape_reports_option(self):
        with pytest.raises(pybamm.OptionError, match=r"option 'particle'"):
            BatteryModelOptions(
                {"particle": (("Fickian diffusion", "a", "b"), "Fickian diffusion")}
            )

    def test_domain_options_resolve_negative_only_shorthand(self):
        items = {"SEI": "constant", "working electrode": "both"}.items()
        assert BatteryModelDomainOptions(items, 0)["SEI"] == "constant"
        assert BatteryModelDomainOptions(items, 1)["SEI"] == "none"
        assert BatteryModelDomainOptions(items, 1).primary["SEI"] == "none"


class TestVaasNormalization:
    """Test the centralized VAAS + surface form policy."""

    def test_vaas_true_with_surface_form_false_is_valid(self):
        """VAAS can be true even with surface_form=false (DFN case)."""
        options = pybamm.BatteryModelOptions(
            {"voltage as a state": "true", "surface form": "false"}
        )
        assert options["voltage as a state"] == "true"
        assert options["surface form"] == "false"

    def test_vaas_false_with_surface_form_algebraic_is_valid(self):
        """surface form algebraic without voltage-as-state is valid."""
        options = pybamm.BatteryModelOptions(
            {"voltage as a state": "false", "surface form": "algebraic"}
        )
        assert options["voltage as a state"] == "false"
        assert options["surface form"] == "algebraic"

    def test_vaas_false_defaults_surface_form_false(self):
        """When VAAS is false and surface form not set, surface form stays false."""
        options = pybamm.BatteryModelOptions({"voltage as a state": "false"})
        assert options["surface form"] == "false"


def repeated_words(name):
    """Return the repeated run in `name`, e.g. "a b a b" in "a b a b c", else None."""
    words = name.split()
    for length in range(1, len(words) // 2 + 1):
        for start in range(len(words) - 2 * length + 1):
            run = words[start : start + length]
            if run == words[start + length : start + 2 * length]:
                return " ".join(run * 2)
    return None


class TestVariableNames:
    @pytest.mark.parametrize(
        ("model_class", "options"),
        [
            (pybamm.lithium_ion.SPM, {}),
            (pybamm.lithium_ion.SPMe, {}),
            (pybamm.lithium_ion.DFN, {}),
            (pybamm.lithium_ion.MPM, {}),
            (pybamm.lithium_ion.NewmanTobias, {}),
            (pybamm.lithium_ion.DFN, {"working electrode": "positive"}),
            (
                pybamm.lithium_ion.DFN,
                {
                    "particle phases": ("2", "1"),
                    "thermal": "x-full",
                    "SEI": "solvent-diffusion limited",
                    "lithium plating": "reversible",
                    "particle mechanics": "swelling and cracking",
                    "cell geometry": "pouch",
                    "surface form": "algebraic",
                    "SEI film resistance": "distributed",
                    "stress-induced diffusion": "true",
                    "total interfacial current density as a state": "true",
                },
            ),
            (pybamm.lead_acid.Full, {}),
            (pybamm.lead_acid.LOQS, {}),
        ],
    )
    def test_no_repeated_words(self, model_class, options):
        # catches names built from a duplicated implicit string concatenation
        model = model_class(options)
        duplicated = {
            name: repeated_words(name)
            for name in model.variables
            if repeated_words(name)
        }
        assert duplicated == {}


class TestOptionHelpers:
    def test_iter_option_leaves_scalar(self):
        assert iter_option_leaves("thermal", "lumped") == [((), "lumped")]

    def test_iter_option_leaves_per_electrode_and_phase(self):
        assert iter_option_leaves(
            "particle mechanics", (("swelling only", "none"), "none")
        ) == [
            (("negative", "primary"), "swelling only"),
            (("negative", "secondary"), "none"),
            (("positive",), "none"),
        ]

    @pytest.mark.parametrize(
        "option, value",
        [
            ("thermal", ("lumped", "lumped")),
            ("particle", ("Fickian diffusion",)),
            ("particle", ("a", "b", "c")),
            ("particle phases", (("1", "1"), "1")),
            ("particle", (("a", "b", "c"), "b")),
            ("particle", ((("a", "b"), "c"), "d")),
            ("particle", ["Fickian diffusion", "Fickian diffusion"]),
        ],
    )
    def test_iter_option_leaves_rejects_bad_shapes(self, option, value):
        with pytest.raises(pybamm.OptionError, match=rf"option '{option}'"):
            iter_option_leaves(option, value)

    def test_resolve_option(self):
        assert resolve_option("particle", "a", "positive") == "a"
        assert resolve_option("particle", ("a", "b"), "positive") == "b"
        assert resolve_option("particle", (("a", "b"), "c"), "negative") == ("a", "b")
        assert (
            resolve_option("particle", (("a", "b"), "c"), "negative", "secondary")
            == "b"
        )
        assert (
            resolve_option("particle", (("a", "b"), "c"), "positive", "secondary")
            == "c"
        )
        assert resolve_option("particle", "a", "negative", "secondary") == "a"

    def test_resolve_option_negative_only_shorthand(self):
        assert resolve_option("SEI", "constant", "negative") == "constant"
        assert resolve_option("SEI", "constant", "positive") == "none"
        assert resolve_option("SEI on cracks", "true", "positive") == "false"
        assert resolve_option("lithium plating", "reversible", "positive") == "none"
        # half cells and explicit tuples are not affected
        assert (
            resolve_option("SEI", "constant", "positive", working_electrode="positive")
            == "constant"
        )
        assert resolve_option("SEI", ("constant", "constant"), "positive") == "constant"
        assert resolve_option("SEI", "none", "positive") == "none"

    def test_validate_option_value(self):
        validate_option_value("thermal", "lumped", ["isothermal", "lumped"])
        validate_option_value("operating mode", lambda t: 1, ["current"])
        validate_option_value("number of MSMR reactions", "3", ["none"])
        with pytest.raises(
            pybamm.OptionError,
            match=r"'bad' is not recognized in option 'particle' at positive",
        ):
            validate_option_value("particle", "bad", ["a"], ("positive",))
        with pytest.raises(pybamm.OptionError, match=r"'0' is not recognized"):
            validate_option_value("number of MSMR reactions", "0", ["none"])

    def test_replace_option_leaf(self):
        assert replace_option_leaf("Axen", "Axen", "new") == "new"
        assert replace_option_leaf((("Axen", "single"), "Axen"), "Axen", "new") == (
            ("new", "single"),
            "new",
        )
        assert replace_option_leaf("single", "Axen", "new") == "single"

    def test_join_electrode_values(self):
        assert join_electrode_values("particle", "a", "a") == "a"
        assert join_electrode_values("particle", "a", "b") == ("a", "b")
        # a scalar SEI in a full cell means negative-only
        assert join_electrode_values("SEI", "constant", "none") == "constant"
        assert join_electrode_values("SEI", "constant", "constant") == (
            "constant",
            "constant",
        )
        assert (
            join_electrode_values(
                "SEI", "constant", "constant", working_electrode="positive"
            )
            == "constant"
        )
        assert join_electrode_values("SEI", "none", "none") == "none"

    def test_active_electrodes(self):
        assert active_electrodes("both") == ("negative", "positive")
        assert active_electrodes("positive") == ("positive",)

    def test_option_values_match(self):
        assert option_values_match("SEI", "constant", ("constant", "none"))
        assert not option_values_match("SEI", "constant", ("constant", "constant"))
        assert option_values_match(
            "SEI", "constant", ("constant", "constant"), working_electrode="positive"
        )
        assert option_values_match("particle", ("a", "a"), "a")
        assert option_values_match("particle", (("a", "a"), "b"), ("a", "b"))
        assert not option_values_match("particle", ("a", "b"), "a")

    def test_dependency_error(self):
        error = dependency_error(
            "lithium plating",
            "partially reversible",
            "SEI",
            "a model other than 'none' (e.g. 'constant')",
            ("negative", "primary"),
        )
        assert isinstance(error, pybamm.OptionError)
        assert str(error) == (
            "Option 'lithium plating' at negative.primary is 'partially reversible', "
            "which requires 'SEI' to be a model other than 'none' (e.g. 'constant')."
        )
        assert str(
            dependency_error("thermal", "x-full", "cell geometry", "'pouch'")
        ) == (
            "Option 'thermal' is 'x-full', which requires 'cell geometry' to be 'pouch'."
        )


@pytest.mark.usefixtures("allow_legacy_defaults")
class TestModelDefaultOptions:
    @pytest.mark.parametrize(
        "model_class, supplied",
        [
            (pybamm.lithium_ion.SPM, {"SEI": "reaction limited"}),
            (pybamm.lithium_ion.SPM, {"intercalation kinetics": "linear"}),
            (pybamm.lithium_ion.SPMe, {"particle size": "distribution"}),
            (pybamm.lithium_ion.MPM, {"SEI": "reaction limited"}),
            (pybamm.lithium_ion.MSMR, {"number of MSMR reactions": ("6", "4")}),
            (pybamm.lithium_ion.NewmanTobias, {"SEI": "constant"}),
            (LithiumMetalDFN, {"thermal": "lumped"}),
            (pybamm.lead_acid.LOQS, {"thermal": "isothermal"}),
            (pybamm.lithium_ion.Yang2017, {"thermal": "lumped"}),
            (pybamm.lithium_ion.BasicDFNHalfCell, {"working electrode": "positive"}),
        ],
    )
    def test_constructor_does_not_mutate_options(self, model_class, supplied):
        # 5801
        snapshot = dict(supplied)
        model_class(supplied)
        assert supplied == snapshot

    def test_spm_options_do_not_leak_into_later_models(self):
        # 5801
        options = {"SEI": "reaction limited"}
        pybamm.lithium_ion.SPM(options)
        dfn = pybamm.lithium_ion.DFN(options)
        assert dfn.options["x-average side reactions"] == "false"

    def test_identity_checks_accept_equivalent_spellings(self):
        # the scalar shorthand resolves to Yang2017's negative-only SEI
        pybamm.lithium_ion.Yang2017({"SEI": "ec reaction limited"})
        model = pybamm.lithium_ion.MSMR(
            {
                "number of MSMR reactions": ("6", "4"),
                "open-circuit potential": ("MSMR", "MSMR"),
            }
        )
        assert model.options.negative["open-circuit potential"] == "MSMR"

    def test_identity_defaults(self):
        assert pybamm.lithium_ion.SPM().options["x-average side reactions"] == "true"
        assert pybamm.lithium_ion.SPMe().options["x-average side reactions"] == "false"
        model = pybamm.lithium_ion.SPM({"intercalation kinetics": "linear"})
        assert model.options["surface form"] == "algebraic"
        model = pybamm.lithium_ion.MPM()
        assert model.options["particle size"] == "distribution"
        assert model.options["surface form"] == "algebraic"
        model = pybamm.lithium_ion.MSMR({"number of MSMR reactions": ("6", "4")})
        assert model.options["particle"] == "MSMR"
        assert (
            pybamm.lithium_ion.NewmanTobias().options["particle"] == "uniform profile"
        )
        assert LithiumMetalDFN().options["working electrode"] == "positive"
        assert pybamm.lead_acid.LOQS().options["particle shape"] == "no particles"
        assert pybamm.lithium_ion.Yang2017().options.negative["SEI"] == (
            "ec reaction limited"
        )
        model = pybamm.lithium_ion.Yang2017({"thermal": "lumped"})
        assert model.options["thermal"] == "lumped"

    @pytest.mark.parametrize(
        "model_class, supplied, match",
        [
            (pybamm.lithium_ion.MPM, {"particle size": "single"}, r"particle size"),
            (pybamm.lithium_ion.MPM, {"surface form": "false"}, r"surface form"),
            (pybamm.lithium_ion.MSMR, {}, r"number of MSMR reactions"),
            (
                pybamm.lithium_ion.MSMR,
                {
                    "number of MSMR reactions": ("6", "4"),
                    "particle": "Fickian diffusion",
                },
                r"'particle' must be 'MSMR'",
            ),
            (
                LithiumMetalDFN,
                {"working electrode": "both"},
                r"working electrode",
            ),
            (pybamm.lead_acid.LOQS, {"particle shape": "spherical"}, r"particle shape"),
            (pybamm.lithium_ion.Yang2017, {"SEI": "constant"}, r"Yang2017"),
            (
                pybamm.lithium_ion.Yang2017,
                {"working electrode": "positive"},
                r"working electrode",
            ),
            (
                pybamm.lithium_ion.BasicDFNHalfCell,
                {"thermal": "lumped"},
                r"BasicDFNHalfCell",
            ),
        ],
    )
    def test_incompatible_identity_overrides(self, model_class, supplied, match):
        with pytest.raises(pybamm.OptionError, match=match):
            model_class(supplied)

    @pytest.mark.parametrize(
        "model_class, supplied",
        [
            (
                pybamm.lithium_ion.DFN,
                {"SEI": "reaction limited", "lithium plating": "partially reversible"},
            ),
            (
                pybamm.lithium_ion.DFN,
                {
                    "particle phases": ("2", "1"),
                    "particle mechanics": (("swelling only", "none"), "none"),
                },
            ),
            (pybamm.lithium_ion.SPM, {"intercalation kinetics": "linear"}),
            (pybamm.lithium_ion.MPM, {}),
            (pybamm.lithium_ion.MSMR, {"number of MSMR reactions": ("6", "4")}),
            (pybamm.lithium_ion.NewmanTobias, {}),
            (LithiumMetalDFN, {}),
            (pybamm.lead_acid.LOQS, {}),
            (pybamm.lithium_ion.Yang2017, {}),
            (pybamm.lithium_ion.BasicDFNHalfCell, {}),
        ],
    )
    def test_reprocessing_options_is_idempotent(self, model_class, supplied):
        model = model_class(supplied)
        assert BatteryModelOptions(dict(model.options)) == model.options
        assert model_class(dict(model.options)).options == model.options
        assert model_class(model.options).options == model.options

    def test_basic_dfn_half_cell_accepts_its_defaults(self):
        model = pybamm.lithium_ion.BasicDFNHalfCell()
        options = model.options
        model.options = dict(options)
        assert model.options == options

    @pytest.mark.parametrize(
        "model_class, match",
        [
            (pybamm.lead_acid.LOQS, r"particle shape"),
            (pybamm.lithium_ion.MPM, r"particle size"),
        ],
    )
    def test_processed_options_are_checked_against_model(self, model_class, match):
        with pytest.raises(pybamm.OptionError, match=match):
            model_class(BatteryModelOptions({}))


LEGACY_DEFAULT_CASES = [
    (
        {"SEI": "constant"},
        {
            "SEI film resistance": "distributed",
            "total interfacial current density as a state": "true",
        },
    ),
    (
        {"lithium plating": "partially reversible"},
        {"SEI": "constant", "SEI film resistance": "none"},
    ),
    (
        {"lithium plating": ("none", "partially reversible")},
        {"SEI": ("none", "constant"), "SEI film resistance": "none"},
    ),
    (
        {"loss of active material": "stress-driven"},
        {"particle mechanics": "swelling only", "stress-induced diffusion": "true"},
    ),
    ({"particle mechanics": "swelling only"}, {"stress-induced diffusion": "true"}),
    (
        {"particle mechanics": ("swelling and cracking", "none")},
        {"stress-induced diffusion": ("true", "false")},
    ),
    ({"particle phases": ("2", "1")}, {"surface form": "algebraic"}),
    ({"dimensionality": 1}, {"cell geometry": "pouch"}),
    ({"thermal": "x-full"}, {"cell geometry": "pouch"}),
    ({"operating mode": "explicit power"}, {"voltage as a state": "true"}),
]


@pytest.mark.usefixtures("allow_legacy_defaults")
class TestLegacyDefaultDeprecation:
    @pytest.mark.parametrize("supplied, fired", LEGACY_DEFAULT_CASES)
    def test_legacy_default_warns_once_with_explicit_options(self, supplied, fired):
        with pytest.warns(pybamm.OptionDefaultDeprecationWarning) as record:
            options = BatteryModelOptions(supplied)
        assert len(record) == 1
        assert str(record[0].message) == base_battery_model._legacy_default_message(
            fired
        )
        # passing the named options reproduces the configuration without a warning
        with warnings.catch_warnings():
            warnings.simplefilter("error", pybamm.OptionDefaultDeprecationWarning)
            explicit = BatteryModelOptions({**supplied, **fired})
        assert explicit == options

    def test_message(self):
        assert base_battery_model._legacy_default_message(
            {"cell geometry": "pouch"}
        ) == (
            "Options were set from other options because they were not given: "
            "{'cell geometry': 'pouch'}. Relying on these defaults is deprecated and "
            "a future release will raise an OptionError instead. Pass these options "
            "explicitly to keep the current behaviour."
        )

    def test_warning_points_at_caller(self):
        with pytest.warns(pybamm.OptionDefaultDeprecationWarning) as record:
            pybamm.lithium_ion.SPM({"SEI": "constant"})
        assert record[0].filename == __file__

    def test_forbidden_legacy_defaults_raise(self, monkeypatch):
        monkeypatch.setattr(base_battery_model, "_FORBID_LEGACY_OPTION_DEFAULTS", True)
        with pytest.raises(pybamm.OptionError, match=r"'cell geometry': 'pouch'"):
            BatteryModelOptions({"dimensionality": 1})

    def test_spm_surface_form_default_warns(self):
        with pytest.warns(pybamm.OptionDefaultDeprecationWarning) as record:
            model = pybamm.lithium_ion.SPM({"intercalation kinetics": "linear"})
        assert [str(r.message) for r in record] == [
            base_battery_model._legacy_default_message({"surface form": "algebraic"})
        ]
        assert model.options["surface form"] == "algebraic"
        with warnings.catch_warnings():
            warnings.simplefilter("error", pybamm.OptionDefaultDeprecationWarning)
            pybamm.lithium_ion.MPM({"intercalation kinetics": "linear"})

    @pytest.mark.parametrize(
        "build_model",
        [
            pybamm.lithium_ion.SPM,
            pybamm.lithium_ion.SPMe,
            pybamm.lithium_ion.DFN,
            pybamm.lithium_ion.MPM,
            pybamm.lithium_ion.NewmanTobias,
            pybamm.lithium_ion.Yang2017,
            pybamm.lithium_ion.BasicSPM,
            pybamm.lithium_ion.BasicDFN,
            pybamm.lithium_ion.BasicDFNHalfCell,
            pybamm.lithium_ion.BasicDFNComposite,
            pybamm.lithium_ion.BasicDFN2D,
            pybamm.lithium_ion.BasicDFNUnstructured,
            pybamm.lead_acid.LOQS,
            pybamm.lead_acid.Full,
            LithiumMetalDFN,
            lambda: pybamm.lithium_ion.MSMR({"number of MSMR reactions": ("6", "4")}),
            lambda: pybamm.lithium_ion.DFN({"working electrode": "positive"}),
        ],
    )
    def test_model_defaults_do_not_warn(self, build_model):
        with warnings.catch_warnings():
            warnings.simplefilter("error", pybamm.OptionDefaultDeprecationWarning)
            build_model()
