#
# Tests for the base battery model class
#

import pytest
from pybamm.models.full_battery_models.base_battery_model import BatteryModelOptions
import pybamm
import io
from contextlib import redirect_stdout
import os

OPTIONS_DICT = {
    "surface form": "differential",
    "loss of active material": "stress-driven",
    "thermal": "x-full",
}

PRINT_OPTIONS_OUTPUT = """\
'calculate discharge energy': 'false' (possible: ['false', 'true'])
'calculate heat source for isothermal models': 'false' (possible: ['false', 'true'])
'cell geometry': 'pouch' (possible: ['arbitrary', 'pouch'])
'contact resistance': 'false' (possible: ['false', 'true'])
'convection': 'none' (possible: ['none', 'uniform transverse', 'full transverse'])
'current collector': 'uniform' (possible: ['uniform', 'potential pair', 'potential pair quite conductive'])
'diffusivity': 'single' (possible: ['single', 'current sigmoid'])
'dimensionality': 0 (possible: [0, 1, 2])
'electrolyte conductivity': 'default' (possible: ['default', 'full', 'leading order', 'composite', 'integrated'])
'exchange-current density': 'single' (possible: ['single', 'current sigmoid'])
'heat of mixing': 'false' (possible: ['false', 'true'])
'hydrolysis': 'false' (possible: ['false', 'true'])
'intercalation kinetics': 'symmetric Butler-Volmer' (possible: ['symmetric Butler-Volmer', 'asymmetric Butler-Volmer', 'linear', 'Marcus', 'Marcus-Hush-Chidsey', 'MSMR'])
'interface utilisation': 'full' (possible: ['full', 'constant', 'current-driven'])
'lithium plating': 'none' (possible: ['none', 'reversible', 'partially reversible', 'irreversible'])
'lithium plating porosity change': 'false' (possible: ['false', 'true'])
'loss of active material': 'stress-driven' (possible: ['none', 'stress-driven', 'reaction-driven', 'current-driven', 'stress and reaction-driven'])
'number of MSMR reactions': 'none' (possible: ['none'])
'open-circuit potential': 'single' (possible: ['single', 'current sigmoid', 'MSMR', 'Wycisk'])
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
"""


class TestBaseBatteryModel:
    def test_process_parameters_and_discretise(self):
        model = pybamm.lithium_ion.SPM()
        # Set up geometry and parameters
        geometry = model.default_geometry
        parameter_values = model.default_parameter_values
        parameter_values.process_geometry(geometry)
        # Set up discretisation
        mesh = pybamm.Mesh(geometry, model.default_submesh_types, model.default_var_pts)
        disc = pybamm.Discretisation(mesh, model.default_spatial_methods)
        # Process expression
        c = (
            pybamm.Parameter("Negative electrode thickness [m]")
            * model.variables["X-averaged negative particle concentration [mol.m-3]"]
        )
        processed_c = model.process_parameters_and_discretise(c, parameter_values, disc)
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

        flux_1 = model.process_parameters_and_discretise(N, parameter_values, disc)
        flux_2 = model.variables["X-averaged negative particle flux [mol.m-2.s-1]"]
        param_flux_2 = parameter_values.process_symbol(flux_2)
        disc_flux_2 = disc.process_symbol(param_flux_2)
        assert flux_1 == disc_flux_2

    def test_summary_variables(self):
        model = pybamm.BaseBatteryModel()
        model.variables["var"] = pybamm.Scalar(1)
        model.summary_variables = ["var"]
        assert model.summary_variables == ["var"]
        with pytest.raises(KeyError, match="No cycling variable defined"):
            model.summary_variables = ["bad var"]

    def test_default_geometry(self):
        model = pybamm.BaseBatteryModel({"dimensionality": 0})
        assert model.default_geometry["current collector"]["z"]["position"] == 1
        model = pybamm.BaseBatteryModel({"dimensionality": 1})
        assert model.default_geometry["current collector"]["z"]["min"] == 0
        model = pybamm.BaseBatteryModel({"dimensionality": 2})
        assert model.default_geometry["current collector"]["y"]["min"] == 0

    def test_default_submesh_types(self):
        model = pybamm.BaseBatteryModel({"dimensionality": 0})
        assert issubclass(
            model.default_submesh_types["current collector"],
            pybamm.SubMesh0D,
        )
        model = pybamm.BaseBatteryModel({"dimensionality": 1})
        assert issubclass(
            model.default_submesh_types["current collector"],
            pybamm.Uniform1DSubMesh,
        )
        model = pybamm.BaseBatteryModel({"dimensionality": 2})
        assert issubclass(
            model.default_submesh_types["current collector"],
            pybamm.ScikitUniform2DSubMesh,
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
        }
        model = pybamm.BaseBatteryModel({"dimensionality": 0})
        assert var_pts == model.default_var_pts

        var_pts.update({"x_n": 10, "x_s": 10, "x_p": 10})
        model = pybamm.BaseBatteryModel({"dimensionality": 2})
        assert var_pts == model.default_var_pts

    def test_default_spatial_methods(self):
        model = pybamm.BaseBatteryModel({"dimensionality": 0})
        assert isinstance(
            model.default_spatial_methods["current collector"],
            pybamm.ZeroDimensionalSpatialMethod,
        )
        model = pybamm.BaseBatteryModel({"dimensionality": 1})
        assert isinstance(
            model.default_spatial_methods["current collector"], pybamm.FiniteVolume
        )
        model = pybamm.BaseBatteryModel({"dimensionality": 2})
        assert isinstance(
            model.default_spatial_methods["current collector"],
            pybamm.ScikitFiniteElement,
        )

    def test_options(self):
        with pytest.raises(pybamm.OptionError, match="Option"):
            pybamm.BaseBatteryModel({"bad option": "bad option"})
        with pytest.raises(pybamm.OptionError, match="current collector model"):
            pybamm.BaseBatteryModel({"current collector": "bad current collector"})
        with pytest.raises(pybamm.OptionError, match="thermal"):
            pybamm.BaseBatteryModel({"thermal": "bad thermal"})
        with pytest.raises(pybamm.OptionError, match="cell geometry"):
            pybamm.BaseBatteryModel({"cell geometry": "bad geometry"})
        with pytest.raises(pybamm.OptionError, match="dimensionality"):
            pybamm.BaseBatteryModel({"dimensionality": 5})
        with pytest.raises(pybamm.OptionError, match="current collector"):
            pybamm.BaseBatteryModel(
                {"dimensionality": 1, "current collector": "bad option"}
            )
        with pytest.raises(pybamm.OptionError, match="1D current collectors"):
            pybamm.BaseBatteryModel(
                {
                    "current collector": "potential pair",
                    "dimensionality": 1,
                    "thermal": "x-full",
                }
            )
        with pytest.raises(pybamm.OptionError, match="2D current collectors"):
            pybamm.BaseBatteryModel(
                {
                    "current collector": "potential pair",
                    "dimensionality": 2,
                    "thermal": "x-full",
                }
            )
        with pytest.raises(pybamm.OptionError, match="surface form"):
            pybamm.BaseBatteryModel({"surface form": "bad surface form"})
        with pytest.raises(pybamm.OptionError, match="convection"):
            pybamm.BaseBatteryModel({"convection": "bad convection"})
        with pytest.raises(
            pybamm.OptionError, match="cannot have transverse convection in 0D model"
        ):
            pybamm.BaseBatteryModel({"convection": "full transverse"})
        with pytest.raises(pybamm.OptionError, match="particle"):
            pybamm.BaseBatteryModel({"particle": "bad particle"})
        with pytest.raises(pybamm.OptionError, match="working electrode"):
            pybamm.BaseBatteryModel({"working electrode": "bad working electrode"})
        with pytest.raises(pybamm.OptionError, match="The 'negative' working"):
            pybamm.BaseBatteryModel({"working electrode": "negative"})
        with pytest.raises(pybamm.OptionError, match="particle shape"):
            pybamm.BaseBatteryModel({"particle shape": "bad particle shape"})
        with pytest.raises(pybamm.OptionError, match="operating mode"):
            pybamm.BaseBatteryModel({"operating mode": "bad operating mode"})
        with pytest.raises(pybamm.OptionError, match="electrolyte conductivity"):
            pybamm.BaseBatteryModel(
                {"electrolyte conductivity": "bad electrolyte conductivity"}
            )

        # SEI options
        with pytest.raises(pybamm.OptionError, match="SEI"):
            pybamm.BaseBatteryModel({"SEI": "bad sei"})
        with pytest.raises(pybamm.OptionError, match="SEI film resistance"):
            pybamm.BaseBatteryModel({"SEI film resistance": "bad SEI film resistance"})
        with pytest.raises(pybamm.OptionError, match="SEI porosity change"):
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
        with pytest.raises(pybamm.OptionError, match="must be 'true'"):
            pybamm.BaseBatteryModel(
                {
                    "SEI film resistance": "distributed",
                    "total interfacial current density as a state": "false",
                }
            )
        with pytest.raises(pybamm.OptionError, match="must be 'true'"):
            pybamm.BaseBatteryModel(
                {
                    "SEI film resistance": "average",
                    "particle phases": "2",
                    "total interfacial current density as a state": "false",
                }
            )

        # loss of active material model
        with pytest.raises(pybamm.OptionError, match="loss of active material"):
            pybamm.BaseBatteryModel({"loss of active material": "bad LAM model"})
        with pytest.raises(pybamm.OptionError, match="loss of active material"):
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
        with pytest.raises(pybamm.OptionError, match="particle mechanics"):
            pybamm.BaseBatteryModel({"particle mechanics": "bad particle cracking"})
        with pytest.raises(pybamm.OptionError, match="particle cracking"):
            pybamm.BaseBatteryModel({"particle cracking": "bad particle cracking"})

        # SEI on cracks
        with pytest.raises(pybamm.OptionError, match="SEI on cracks"):
            pybamm.BaseBatteryModel({"SEI on cracks": "bad SEI on cracks"})
        with pytest.raises(pybamm.OptionError, match="'SEI on cracks' is 'true'"):
            pybamm.BaseBatteryModel(
                {"SEI on cracks": "true", "particle mechanics": "swelling only"}
            )

        # plating model
        with pytest.raises(pybamm.OptionError, match="lithium plating"):
            pybamm.BaseBatteryModel({"lithium plating": "bad plating"})
        with pytest.raises(pybamm.OptionError, match="lithium plating porosity change"):
            pybamm.BaseBatteryModel(
                {
                    "lithium plating porosity change": "bad lithium "
                    "plating porosity change"
                }
            )

        # contact resistance
        with pytest.raises(pybamm.OptionError, match="contact resistance"):
            pybamm.BaseBatteryModel({"contact resistance": "bad contact resistance"})
        with pytest.raises(NotImplementedError, match="Contact resistance not yet"):
            pybamm.BaseBatteryModel(
                {
                    "contact resistance": "true",
                    "operating mode": "explicit power",
                }
            )
        with pytest.raises(NotImplementedError, match="Contact resistance not yet"):
            pybamm.BaseBatteryModel(
                {
                    "contact resistance": "true",
                    "operating mode": "explicit resistance",
                }
            )

        # stress-induced diffusion
        with pytest.raises(pybamm.OptionError, match="cannot have stress"):
            pybamm.BaseBatteryModel({"stress-induced diffusion": "true"})

        # hydrolysis
        with pytest.raises(pybamm.OptionError, match="surface formulation"):
            pybamm.lead_acid.LOQS({"hydrolysis": "true", "surface form": "false"})

        # timescale
        with pytest.raises(pybamm.OptionError, match="timescale"):
            pybamm.BaseBatteryModel({"timescale": "bad timescale"})

        # thermal x-lumped
        with pytest.raises(pybamm.OptionError, match="x-lumped"):
            pybamm.lithium_ion.BaseModel(
                {"cell geometry": "arbitrary", "thermal": "x-lumped"}
            )

        # thermal half-cell
        with pytest.raises(pybamm.OptionError, match="X-full"):
            pybamm.BaseBatteryModel(
                {"thermal": "x-full", "working electrode": "positive"}
            )
        with pytest.raises(pybamm.OptionError, match="X-lumped"):
            pybamm.BaseBatteryModel(
                {
                    "dimensionality": 2,
                    "thermal": "x-lumped",
                    "working electrode": "positive",
                }
            )

        # thermal heat of mixing
        with pytest.raises(NotImplementedError, match="Heat of mixing"):
            pybamm.BaseBatteryModel(
                {
                    "heat of mixing": "true",
                    "particle size": "distribution",
                }
            )

        # surface thermal model
        with pytest.raises(pybamm.OptionError, match="surface temperature"):
            pybamm.BaseBatteryModel(
                {"surface temperature": "lumped", "thermal": "x-full"}
            )

        # phases
        with pytest.raises(pybamm.OptionError, match="multiple particle phases"):
            pybamm.BaseBatteryModel({"particle phases": "2", "surface form": "false"})

        # msmr
        with pytest.raises(pybamm.OptionError, match="MSMR"):
            pybamm.BaseBatteryModel({"open-circuit potential": "MSMR"})
        with pytest.raises(pybamm.OptionError, match="MSMR"):
            pybamm.BaseBatteryModel({"particle": "MSMR"})
        with pytest.raises(pybamm.OptionError, match="MSMR"):
            pybamm.BaseBatteryModel({"intercalation kinetics": "MSMR"})
        with pytest.raises(pybamm.OptionError, match="MSMR"):
            pybamm.BaseBatteryModel(
                {"open-circuit potential": "MSMR", "particle": "MSMR"}
            )
        with pytest.raises(pybamm.OptionError, match="MSMR"):
            pybamm.BaseBatteryModel(
                {"open-circuit potential": "MSMR", "intercalation kinetics": "MSMR"}
            )
        with pytest.raises(pybamm.OptionError, match="MSMR"):
            pybamm.BaseBatteryModel(
                {"particle": "MSMR", "intercalation kinetics": "MSMR"}
            )
        with pytest.raises(pybamm.OptionError, match="MSMR"):
            pybamm.BaseBatteryModel(
                {
                    "open-circuit potential": "MSMR",
                    "particle": "MSMR",
                    "intercalation kinetics": "MSMR",
                    "number of MSMR reactions": "1.5",
                }
            )

    def test_build_twice(self):
        model = pybamm.lithium_ion.SPM()  # need to pick a model to set vars and build
        with pytest.raises(pybamm.ModelError, match="Model already built"):
            model.build_model()

    def test_get_coupled_variables(self):
        model = pybamm.lithium_ion.BaseModel()
        model.submodels["current collector"] = pybamm.current_collector.Uniform(
            model.param
        )
        with pytest.raises(pybamm.ModelError, match="Missing variable"):
            model.build_model()

    def test_default_solver(self):
        model = pybamm.BaseBatteryModel()
        assert isinstance(model.default_solver, pybamm.CasadiSolver)

        # check that default_solver gives you a new solver, not an internal object
        solver = model.default_solver
        solver = pybamm.BaseModel()
        assert isinstance(model.default_solver, pybamm.CasadiSolver)
        assert isinstance(solver, pybamm.BaseModel)

        # check that adding algebraic variables gives algebraic solver
        a = pybamm.Variable("a")
        model.algebraic = {a: a - 1}
        assert isinstance(model.default_solver, pybamm.CasadiAlgebraicSolver)

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
        model = pybamm.lithium_ion.SPM()
        geometry = model.default_geometry
        param = model.default_parameter_values
        param.process_model(model)
        param.process_geometry(geometry)
        mesh = pybamm.Mesh(geometry, model.default_submesh_types, model.default_var_pts)
        disc = pybamm.Discretisation(mesh, model.default_spatial_methods)
        disc.process_model(model)

        # save model
        model.save_model(
            filename="test_base_battery_model", mesh=mesh, variables=model.variables
        )

        # raises error if variables are saved without mesh
        with pytest.raises(ValueError):
            model.save_model(
                filename="test_base_battery_model", variables=model.variables
            )

        os.remove("test_base_battery_model.json")

    def test_voltage_as_state(self):
        model = pybamm.lithium_ion.SPM({"voltage as a state": "true"})
        assert model.options["voltage as a state"] == "true"
        assert isinstance(model.variables["Voltage [V]"], pybamm.Variable)

        model = pybamm.lithium_ion.SPM(
            {"voltage as a state": "true", "operating mode": "voltage"}
        )
        assert model.options["voltage as a state"] == "true"
        assert isinstance(model.variables["Voltage [V]"], pybamm.Variable)


class TestOptions:
    def test_print_options(self):
        with io.StringIO() as buffer, redirect_stdout(buffer):
            BatteryModelOptions(OPTIONS_DICT).print_options()
            output = buffer.getvalue()

        assert output == PRINT_OPTIONS_OUTPUT

    def test_option_phases(self):
        options = BatteryModelOptions({})
        assert options.phases == {"negative": ["primary"], "positive": ["primary"]}
        options = BatteryModelOptions({"particle phases": ("1", "2")})
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
            {"particle mechanics": (("swelling only", "swelling and cracking"), "none")}
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
