#
# Tests for the base battery model class
#
from pybamm.models.full_battery_models.base_battery_model import BatteryModelOptions
import pybamm
import unittest
import io
from contextlib import redirect_stdout

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
'dimensionality': 0 (possible: [0, 1, 2])
'electrolyte conductivity': 'default' (possible: ['default', 'full', 'leading order', 'composite', 'integrated'])
'hydrolysis': 'false' (possible: ['false', 'true'])
'intercalation kinetics': 'symmetric Butler-Volmer' (possible: ['symmetric Butler-Volmer', 'asymmetric Butler-Volmer', 'linear', 'Marcus', 'Marcus-Hush-Chidsey'])
'interface utilisation': 'full' (possible: ['full', 'constant', 'current-driven'])
'lithium plating': 'none' (possible: ['none', 'reversible', 'partially reversible', 'irreversible'])
'lithium plating porosity change': 'false' (possible: ['false', 'true'])
'loss of active material': 'stress-driven' (possible: ['none', 'stress-driven', 'reaction-driven', 'stress and reaction-driven'])
'open circuit potential': 'single' (possible: ['single', 'current sigmoid'])
'operating mode': 'current' (possible: ['current', 'voltage', 'power', 'differential power', 'explicit power', 'resistance', 'differential resistance', 'explicit resistance', 'CCCV'])
'particle': 'Fickian diffusion' (possible: ['Fickian diffusion', 'fast diffusion', 'uniform profile', 'quadratic profile', 'quartic profile'])
'particle mechanics': 'swelling only' (possible: ['none', 'swelling only', 'swelling and cracking'])
'particle phases': '1' (possible: ['1', '2'])
'particle shape': 'spherical' (possible: ['spherical', 'no particles'])
'particle size': 'single' (possible: ['single', 'distribution'])
'SEI': 'none' (possible: ['none', 'constant', 'reaction limited', 'reaction limited (asymmetric)', 'solvent-diffusion limited', 'electron-migration limited', 'interstitial-diffusion limited', 'ec reaction limited', 'ec reaction limited (asymmetric)'])
'SEI film resistance': 'none' (possible: ['none', 'distributed', 'average'])
'SEI on cracks': 'false' (possible: ['false', 'true'])
'SEI porosity change': 'false' (possible: ['false', 'true'])
'stress-induced diffusion': 'true' (possible: ['false', 'true'])
'surface form': 'differential' (possible: ['false', 'differential', 'algebraic'])
'thermal': 'x-full' (possible: ['isothermal', 'lumped', 'x-lumped', 'x-full'])
'total interfacial current density as a state': 'false' (possible: ['false', 'true'])
'working electrode': 'both' (possible: ['both', 'negative', 'positive'])
'x-average side reactions': 'false' (possible: ['false', 'true'])
'timescale': 'default'
"""  # noqa: E501


class TestBaseBatteryModel(unittest.TestCase):
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
        c = pybamm.Parameter("Negative electrode thickness [m]") * pybamm.Variable(
            "X-averaged negative particle concentration",
            domain="negative particle",
            auxiliary_domains={"secondary": "current collector"},
        )
        processed_c = model.process_parameters_and_discretise(c, parameter_values, disc)
        self.assertIsInstance(processed_c, pybamm.Multiplication)
        self.assertIsInstance(processed_c.left, pybamm.Scalar)
        self.assertIsInstance(processed_c.right, pybamm.StateVector)
        # Process flux manually and check result against flux computed in particle
        # submodel
        c_n = model.variables["X-averaged negative particle concentration"]
        T = pybamm.PrimaryBroadcast(
            model.variables["X-averaged negative electrode temperature"],
            ["negative particle"],
        )
        D = model.param.n.prim.D(c_n, T)
        N = -D * pybamm.grad(c_n)

        flux_1 = model.process_parameters_and_discretise(N, parameter_values, disc)
        flux_2 = model.variables["X-averaged negative particle flux"]
        param_flux_2 = parameter_values.process_symbol(flux_2)
        disc_flux_2 = disc.process_symbol(param_flux_2)
        self.assertEqual(flux_1, disc_flux_2)

    def test_summary_variables(self):
        model = pybamm.BaseBatteryModel()
        model.variables["var"] = pybamm.Scalar(1)
        model.summary_variables = ["var"]
        self.assertEqual(model.summary_variables, ["var"])
        with self.assertRaisesRegex(KeyError, "No cycling variable defined"):
            model.summary_variables = ["bad var"]

    def test_timescale_lengthscale_errors(self):
        model = pybamm.BaseBatteryModel()
        with self.assertRaisesRegex(NotImplementedError, "Timescale cannot be"):
            model.timescale = 1
        with self.assertRaisesRegex(NotImplementedError, "Length scales cannot be"):
            model.length_scales = {}

    def test_default_geometry(self):
        model = pybamm.BaseBatteryModel({"dimensionality": 0})
        self.assertEqual(
            model.default_geometry["current collector"]["z"]["position"], 1
        )
        model = pybamm.BaseBatteryModel({"dimensionality": 1})
        self.assertEqual(model.default_geometry["current collector"]["z"]["min"], 0)
        model = pybamm.BaseBatteryModel({"dimensionality": 2})
        self.assertEqual(model.default_geometry["current collector"]["y"]["min"], 0)

    def test_default_submesh_types(self):
        model = pybamm.BaseBatteryModel({"dimensionality": 0})
        self.assertTrue(
            issubclass(
                model.default_submesh_types["current collector"],
                pybamm.SubMesh0D,
            )
        )
        model = pybamm.BaseBatteryModel({"dimensionality": 1})
        self.assertTrue(
            issubclass(
                model.default_submesh_types["current collector"],
                pybamm.Uniform1DSubMesh,
            )
        )
        model = pybamm.BaseBatteryModel({"dimensionality": 2})
        self.assertTrue(
            issubclass(
                model.default_submesh_types["current collector"],
                pybamm.ScikitUniform2DSubMesh,
            )
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
        self.assertDictEqual(var_pts, model.default_var_pts)

        var_pts.update({"x_n": 10, "x_s": 10, "x_p": 10})
        model = pybamm.BaseBatteryModel({"dimensionality": 2})
        self.assertDictEqual(var_pts, model.default_var_pts)

    def test_default_spatial_methods(self):
        model = pybamm.BaseBatteryModel({"dimensionality": 0})
        self.assertIsInstance(
            model.default_spatial_methods["current collector"],
            pybamm.ZeroDimensionalSpatialMethod,
        )
        model = pybamm.BaseBatteryModel({"dimensionality": 1})
        self.assertIsInstance(
            model.default_spatial_methods["current collector"], pybamm.FiniteVolume
        )
        model = pybamm.BaseBatteryModel({"dimensionality": 2})
        self.assertIsInstance(
            model.default_spatial_methods["current collector"],
            pybamm.ScikitFiniteElement,
        )

    def test_options(self):
        with self.assertRaisesRegex(pybamm.OptionError, "Option"):
            pybamm.BaseBatteryModel({"bad option": "bad option"})
        with self.assertRaisesRegex(pybamm.OptionError, "current collector model"):
            pybamm.BaseBatteryModel({"current collector": "bad current collector"})
        with self.assertRaisesRegex(pybamm.OptionError, "thermal"):
            pybamm.BaseBatteryModel({"thermal": "bad thermal"})
        with self.assertRaisesRegex(pybamm.OptionError, "cell geometry"):
            pybamm.BaseBatteryModel({"cell geometry": "bad geometry"})
        with self.assertRaisesRegex(pybamm.OptionError, "dimensionality"):
            pybamm.BaseBatteryModel({"dimensionality": 5})
        with self.assertRaisesRegex(pybamm.OptionError, "current collector"):
            pybamm.BaseBatteryModel(
                {"dimensionality": 1, "current collector": "bad option"}
            )
        with self.assertRaisesRegex(pybamm.OptionError, "1D current collectors"):
            pybamm.BaseBatteryModel(
                {
                    "current collector": "potential pair",
                    "dimensionality": 1,
                    "thermal": "x-full",
                }
            )
        with self.assertRaisesRegex(pybamm.OptionError, "2D current collectors"):
            pybamm.BaseBatteryModel(
                {
                    "current collector": "potential pair",
                    "dimensionality": 2,
                    "thermal": "x-full",
                }
            )
        with self.assertRaisesRegex(pybamm.OptionError, "surface form"):
            pybamm.BaseBatteryModel({"surface form": "bad surface form"})
        with self.assertRaisesRegex(pybamm.OptionError, "convection"):
            pybamm.BaseBatteryModel({"convection": "bad convection"})
        with self.assertRaisesRegex(
            pybamm.OptionError, "cannot have transverse convection in 0D model"
        ):
            pybamm.BaseBatteryModel({"convection": "full transverse"})
        with self.assertRaisesRegex(pybamm.OptionError, "particle"):
            pybamm.BaseBatteryModel({"particle": "bad particle"})
        with self.assertRaisesRegex(pybamm.OptionError, "The 'fast diffusion'"):
            pybamm.BaseBatteryModel({"particle": "fast diffusion"})
        with self.assertRaisesRegex(pybamm.OptionError, "particle shape"):
            pybamm.BaseBatteryModel({"particle shape": "bad particle shape"})
        with self.assertRaisesRegex(pybamm.OptionError, "operating mode"):
            pybamm.BaseBatteryModel({"operating mode": "bad operating mode"})
        with self.assertRaisesRegex(pybamm.OptionError, "electrolyte conductivity"):
            pybamm.BaseBatteryModel(
                {"electrolyte conductivity": "bad electrolyte conductivity"}
            )

        # SEI options
        with self.assertRaisesRegex(pybamm.OptionError, "SEI"):
            pybamm.BaseBatteryModel({"SEI": "bad sei"})
        with self.assertRaisesRegex(pybamm.OptionError, "SEI film resistance"):
            pybamm.BaseBatteryModel({"SEI film resistance": "bad SEI film resistance"})
        with self.assertRaisesRegex(pybamm.OptionError, "SEI porosity change"):
            pybamm.BaseBatteryModel({"SEI porosity change": "bad SEI porosity change"})
        with self.assertRaisesRegex(
            pybamm.OptionError, "SEI porosity change must now be given in string format"
        ):
            pybamm.BaseBatteryModel({"SEI porosity change": True})
        # changing defaults based on other options
        model = pybamm.BaseBatteryModel()
        self.assertEqual(model.options["SEI film resistance"], "none")
        model = pybamm.BaseBatteryModel({"SEI": "constant"})
        self.assertEqual(model.options["SEI film resistance"], "distributed")
        self.assertEqual(
            model.options["total interfacial current density as a state"], "true"
        )
        model = pybamm.BaseBatteryModel(
            {"SEI film resistance": "average", "particle phases": "2"}
        )
        self.assertEqual(
            model.options["total interfacial current density as a state"], "true"
        )
        with self.assertRaisesRegex(pybamm.OptionError, "must be 'true'"):
            pybamm.BaseBatteryModel(
                {
                    "SEI film resistance": "distributed",
                    "total interfacial current density as a state": "false",
                }
            )
        with self.assertRaisesRegex(pybamm.OptionError, "must be 'true'"):
            pybamm.BaseBatteryModel(
                {
                    "SEI film resistance": "average",
                    "particle phases": "2",
                    "total interfacial current density as a state": "false",
                }
            )

        # loss of active material model
        with self.assertRaisesRegex(pybamm.OptionError, "loss of active material"):
            pybamm.BaseBatteryModel({"loss of active material": "bad LAM model"})
        with self.assertRaisesRegex(pybamm.OptionError, "loss of active material"):
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
        self.assertEqual(
            model.options["particle mechanics"],
            ("swelling and cracking", "swelling only"),
        )
        self.assertEqual(model.options["stress-induced diffusion"], "true")

        # crack model
        with self.assertRaisesRegex(pybamm.OptionError, "particle mechanics"):
            pybamm.BaseBatteryModel({"particle mechanics": "bad particle cracking"})
        with self.assertRaisesRegex(pybamm.OptionError, "particle cracking"):
            pybamm.BaseBatteryModel({"particle cracking": "bad particle cracking"})

        # SEI on cracks
        with self.assertRaisesRegex(pybamm.OptionError, "SEI on cracks"):
            pybamm.BaseBatteryModel({"SEI on cracks": "bad SEI on cracks"})
        with self.assertRaisesRegex(NotImplementedError, "SEI on cracks not yet"):
            pybamm.BaseBatteryModel(
                {
                    "SEI on cracks": "true",
                    "working electrode": "positive",
                }
            )

        # plating model
        with self.assertRaisesRegex(pybamm.OptionError, "lithium plating"):
            pybamm.BaseBatteryModel({"lithium plating": "bad plating"})
        with self.assertRaisesRegex(
            pybamm.OptionError, "lithium plating porosity change"
        ):
            pybamm.BaseBatteryModel(
                {
                    "lithium plating porosity change": "bad lithium "
                    "plating porosity change"
                }
            )

        # contact resistance
        with self.assertRaisesRegex(pybamm.OptionError, "contact resistance"):
            pybamm.BaseBatteryModel({"contact resistance": "bad contact resistance"})
        with self.assertRaisesRegex(NotImplementedError, "Contact resistance not yet"):
            pybamm.BaseBatteryModel(
                {
                    "contact resistance": "true",
                    "operating mode": "explicit power",
                }
            )
        with self.assertRaisesRegex(NotImplementedError, "Contact resistance not yet"):
            pybamm.BaseBatteryModel(
                {
                    "contact resistance": "true",
                    "operating mode": "explicit resistance",
                }
            )

        # stress-induced diffusion
        with self.assertRaisesRegex(pybamm.OptionError, "cannot have stress"):
            pybamm.BaseBatteryModel({"stress-induced diffusion": "true"})

        # hydrolysis
        with self.assertRaisesRegex(pybamm.OptionError, "surface formulation"):
            pybamm.lead_acid.LOQS({"hydrolysis": "true", "surface form": "false"})

        # timescale
        with self.assertRaisesRegex(pybamm.OptionError, "timescale"):
            pybamm.BaseBatteryModel({"timescale": "bad timescale"})

        # thermal x-lumped
        with self.assertRaisesRegex(pybamm.OptionError, "x-lumped"):
            pybamm.lithium_ion.BaseModel(
                {"cell geometry": "arbitrary", "thermal": "x-lumped"}
            )

        # thermal half-cell
        with self.assertRaisesRegex(pybamm.OptionError, "X-full"):
            pybamm.BaseBatteryModel(
                {"thermal": "x-full", "working electrode": "positive"}
            )
        with self.assertRaisesRegex(pybamm.OptionError, "X-lumped"):
            pybamm.BaseBatteryModel(
                {
                    "dimensionality": 2,
                    "thermal": "x-lumped",
                    "working electrode": "positive",
                }
            )

        # phases
        with self.assertRaisesRegex(pybamm.OptionError, "multiple particle phases"):
            pybamm.BaseBatteryModel({"particle phases": "2", "surface form": "false"})

    def test_build_twice(self):
        model = pybamm.lithium_ion.SPM()  # need to pick a model to set vars and build
        with self.assertRaisesRegex(pybamm.ModelError, "Model already built"):
            model.build_model()

    def test_get_coupled_variables(self):
        model = pybamm.lithium_ion.BaseModel()
        model.submodels["current collector"] = pybamm.current_collector.Uniform(
            model.param
        )
        with self.assertRaisesRegex(pybamm.ModelError, "Missing variable"):
            model.build_model()

    def test_default_solver(self):
        model = pybamm.BaseBatteryModel()
        self.assertIsInstance(model.default_solver, pybamm.CasadiSolver)

        # check that default_solver gives you a new solver, not an internal object
        solver = model.default_solver
        solver = pybamm.BaseModel()
        self.assertIsInstance(model.default_solver, pybamm.CasadiSolver)
        self.assertIsInstance(solver, pybamm.BaseModel)

        # check that adding algebraic variables gives algebraic solver
        a = pybamm.Variable("a")
        model.algebraic = {a: a - 1}
        self.assertIsInstance(model.default_solver, pybamm.CasadiAlgebraicSolver)

    def test_timescale(self):
        model = pybamm.BaseModel()
        self.assertEqual(model.timescale.evaluate(), 1)

    def test_option_type(self):
        # no entry gets default options
        model = pybamm.BaseBatteryModel()
        self.assertIsInstance(model.options, pybamm.BatteryModelOptions)

        # dict options get converted to BatteryModelOptions
        model = pybamm.BaseBatteryModel({"thermal": "isothermal"})
        self.assertIsInstance(model.options, pybamm.BatteryModelOptions)

        # special dict types are not changed
        options = pybamm.FuzzyDict({"thermal": "isothermal"})
        model = pybamm.BaseBatteryModel(options)
        self.assertEqual(model.options, options)


class TestOptions(unittest.TestCase):
    def test_print_options(self):
        with io.StringIO() as buffer, redirect_stdout(buffer):
            BatteryModelOptions(OPTIONS_DICT).print_options()
            output = buffer.getvalue()
        self.assertEqual(output, PRINT_OPTIONS_OUTPUT)

    def test_option_phases(self):
        options = BatteryModelOptions({})
        self.assertEqual(
            options.phases, {"negative": ["primary"], "positive": ["primary"]}
        )
        options = BatteryModelOptions({"particle phases": ("1", "2")})
        self.assertEqual(
            options.phases,
            {"negative": ["primary"], "positive": ["primary", "secondary"]},
        )

    def test_domain_options(self):
        options = BatteryModelOptions(
            {"particle": ("Fickian diffusion", "quadratic profile")}
        )
        self.assertEqual(options.negative["particle"], "Fickian diffusion")
        self.assertEqual(options.positive["particle"], "quadratic profile")
        # something that is the same in both domains
        self.assertEqual(options.negative["thermal"], "isothermal")
        self.assertEqual(options.positive["thermal"], "isothermal")

    def test_domain_phase_options(self):
        options = BatteryModelOptions(
            {"particle mechanics": (("swelling only", "swelling and cracking"), "none")}
        )
        self.assertEqual(
            options.negative["particle mechanics"],
            ("swelling only", "swelling and cracking"),
        )
        self.assertEqual(
            options.negative.primary["particle mechanics"], "swelling only"
        )
        self.assertEqual(
            options.negative.secondary["particle mechanics"], "swelling and cracking"
        )
        self.assertEqual(options.positive["particle mechanics"], "none")
        self.assertEqual(options.positive.primary["particle mechanics"], "none")
        self.assertEqual(options.positive.secondary["particle mechanics"], "none")

    def test_whole_cell_domains(self):
        options = BatteryModelOptions({"working electrode": "positive"})
        self.assertEqual(
            options.whole_cell_domains, ["separator", "positive electrode"]
        )

        options = BatteryModelOptions({"working electrode": "negative"})
        self.assertEqual(
            options.whole_cell_domains, ["negative electrode", "separator"]
        )

        options = BatteryModelOptions({})
        self.assertEqual(
            options.whole_cell_domains,
            ["negative electrode", "separator", "positive electrode"],
        )


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
