#
# Tests the citations class.
#
import pybamm
import unittest


class TestCitations(unittest.TestCase):
    def test_citations(self):
        citations = pybamm.citations
        # Default papers should be in both _all_citations dict and in the papers to cite
        self.assertIn("Sulzer2021", citations._all_citations.keys())
        self.assertIn("Sulzer2021", citations._papers_to_cite)
        self.assertIn("Harris2020", citations._papers_to_cite)
        # Non-default papers should only be in the _all_citations dict
        self.assertIn("Sulzer2019physical", citations._all_citations.keys())
        self.assertNotIn("Sulzer2019physical", citations._papers_to_cite)

        # test key error
        with self.assertRaises(KeyError):
            citations.register("not a citation")

    def test_print_citations(self):
        pybamm.citations._reset()
        pybamm.print_citations("test_citations.txt", "text")
        pybamm.print_citations("test_citations.txt", "bibtex")
        pybamm.citations._papers_to_cite = set()
        pybamm.print_citations()
        with self.assertRaisesRegex(pybamm.OptionError, "'text' or 'bibtex'"):
            pybamm.print_citations("test_citations.txt", "bad format")

    def test_andersson_2019(self):
        citations = pybamm.citations
        citations._reset()
        self.assertNotIn("Andersson2019", citations._papers_to_cite)
        pybamm.CasadiConverter()
        self.assertIn("Andersson2019", citations._papers_to_cite)

    def test_marquis_2019(self):
        # Test that calling relevant bits of code adds the right paper to citations
        citations = pybamm.citations

        citations._reset()
        self.assertNotIn("Marquis2019", citations._papers_to_cite)
        pybamm.lithium_ion.SPM(build=False)
        self.assertIn("Marquis2019", citations._papers_to_cite)

        citations._reset()
        pybamm.lithium_ion.SPMe(build=False)
        self.assertIn("Marquis2019", citations._papers_to_cite)

    def test_doyle_1993(self):
        citations = pybamm.citations
        citations._reset()
        self.assertNotIn("Doyle1993", citations._papers_to_cite)
        pybamm.lithium_ion.DFN(build=False)
        self.assertIn("Doyle1993", citations._papers_to_cite)

    def test_sulzer_2019(self):
        # Test that calling relevant bits of code adds the right paper to citations
        citations = pybamm.citations

        citations._reset()
        self.assertNotIn("Sulzer2019asymptotic", citations._papers_to_cite)
        pybamm.lead_acid.LOQS(build=False)
        self.assertIn("Sulzer2019asymptotic", citations._papers_to_cite)

        citations._reset()
        pybamm.lead_acid.FOQS(build=False)
        self.assertIn("Sulzer2019asymptotic", citations._papers_to_cite)

        citations._reset()
        pybamm.lead_acid.Composite(build=False)
        self.assertIn("Sulzer2019asymptotic", citations._papers_to_cite)

        citations._reset()
        pybamm.lead_acid.Full(build=False)
        self.assertIn("Sulzer2019physical", citations._papers_to_cite)

    def test_timms_2021(self):
        # Test that calling relevant bits of code adds the right paper to citations
        citations = pybamm.citations

        citations._reset()
        self.assertNotIn("Timms2021", citations._papers_to_cite)
        pybamm.current_collector.BasePotentialPair(param=None)
        self.assertIn("Timms2021", citations._papers_to_cite)

        citations._reset()
        self.assertNotIn("Timms2021", citations._papers_to_cite)
        pybamm.current_collector.EffectiveResistance()
        self.assertIn("Timms2021", citations._papers_to_cite)

        citations._reset()
        self.assertNotIn("Timms2021", citations._papers_to_cite)
        pybamm.current_collector.AlternativeEffectiveResistance2D()
        self.assertIn("Timms2021", citations._papers_to_cite)

        citations._reset()
        self.assertNotIn("Timms2021", citations._papers_to_cite)
        pybamm.thermal.pouch_cell.CurrentCollector1D(param=None)
        self.assertIn("Timms2021", citations._papers_to_cite)

        citations._reset()
        self.assertNotIn("Timms2021", citations._papers_to_cite)
        pybamm.thermal.pouch_cell.CurrentCollector2D(param=None)
        self.assertIn("Timms2021", citations._papers_to_cite)

        citations._reset()
        self.assertNotIn("Timms2021", citations._papers_to_cite)
        pybamm.thermal.Lumped(param=None)
        self.assertIn("Timms2021", citations._papers_to_cite)

        citations._reset()
        self.assertNotIn("Timms2021", citations._papers_to_cite)
        pybamm.thermal.OneDimensionalX(param=None)
        self.assertIn("Timms2021", citations._papers_to_cite)

    def test_subramanian_2005(self):
        # Test that calling relevant bits of code adds the right paper to citations
        citations = pybamm.citations

        citations._reset()
        self.assertNotIn("Subramanian2005", citations._papers_to_cite)
        pybamm.particle.XAveragedPolynomialProfile(
            None, "Negative", {"particle": "quadratic profile"}
        )
        self.assertIn("Subramanian2005", citations._papers_to_cite)

        citations._reset()
        self.assertNotIn("Subramanian2005", citations._papers_to_cite)
        pybamm.particle.PolynomialProfile(
            None, "Negative", {"particle": "quadratic profile"}
        )
        self.assertIn("Subramanian2005", citations._papers_to_cite)

    def test_brosaplanella_2021(self):
        # Test that calling relevant bits of code adds the right paper to citations
        citations = pybamm.citations

        citations._reset()
        self.assertNotIn("BrosaPlanella2021", citations._papers_to_cite)
        pybamm.electrolyte_conductivity.Integrated(None)
        self.assertIn("BrosaPlanella2021", citations._papers_to_cite)

    def test_newman_tobias(self):
        # Test that calling relevant bits of code adds the right paper to citations
        citations = pybamm.citations

        citations._reset()
        self.assertNotIn("Newman1962", citations._papers_to_cite)
        self.assertNotIn("Chu2020", citations._papers_to_cite)
        pybamm.lithium_ion.NewmanTobias()
        self.assertIn("Newman1962", citations._papers_to_cite)
        self.assertIn("Chu2020", citations._papers_to_cite)

    def test_scikit_fem(self):
        citations = pybamm.citations

        citations._reset()
        self.assertNotIn("Gustafsson2020", citations._papers_to_cite)
        pybamm.ScikitFiniteElement()
        self.assertIn("Gustafsson2020", citations._papers_to_cite)

    def test_reniers_2019(self):
        # Test that calling relevant bits of code adds the right paper to citations
        citations = pybamm.citations

        citations._reset()
        self.assertNotIn("Reniers2019", citations._papers_to_cite)
        pybamm.active_material.LossActiveMaterial(None, None, None, True)
        self.assertIn("Reniers2019", citations._papers_to_cite)

    def test_mohtat_2019(self):
        citations = pybamm.citations

        citations._reset()
        self.assertNotIn("Mohtat2019", citations._papers_to_cite)
        pybamm.lithium_ion.ElectrodeSOHx100()
        self.assertIn("Mohtat2019", citations._papers_to_cite)

        citations._reset()
        self.assertNotIn("Mohtat2019", citations._papers_to_cite)
        pybamm.lithium_ion.ElectrodeSOHx0()
        self.assertIn("Mohtat2019", citations._papers_to_cite)

    def test_mohtat_2021(self):
        citations = pybamm.citations

        citations._reset()
        self.assertNotIn("Mohtat2021", citations._papers_to_cite)
        pybamm.external_circuit.CCCVFunctionControl(None, None)
        self.assertIn("Mohtat2021", citations._papers_to_cite)

    def test_sripad_2020(self):
        citations = pybamm.citations

        citations._reset()
        self.assertNotIn("Sripad2020", citations._papers_to_cite)
        pybamm.kinetics.Marcus(None, None, None, None)
        self.assertIn("Sripad2020", citations._papers_to_cite)

        citations._reset()
        self.assertNotIn("Sripad2020", citations._papers_to_cite)
        pybamm.kinetics.MarcusHushChidsey(None, None, None, None)
        self.assertIn("Sripad2020", citations._papers_to_cite)

    def test_parameter_citations(self):
        citations = pybamm.citations

        citations._reset()
        pybamm.ParameterValues("Chen2020")
        self.assertIn("Chen2020", citations._papers_to_cite)

        citations._reset()
        pybamm.ParameterValues("NCA_Kim2011")
        self.assertIn("Kim2011", citations._papers_to_cite)

        citations._reset()
        pybamm.ParameterValues("Marquis2019")
        self.assertIn("Marquis2019", citations._papers_to_cite)

        citations._reset()
        pybamm.ParameterValues("Sulzer2019")
        self.assertIn("Sulzer2019physical", citations._papers_to_cite)

        citations._reset()
        pybamm.ParameterValues("Ecker2015")
        self.assertIn("Ecker2015i", citations._papers_to_cite)
        self.assertIn("Ecker2015ii", citations._papers_to_cite)
        self.assertIn("Zhao2018", citations._papers_to_cite)
        self.assertIn("Hales2019", citations._papers_to_cite)
        self.assertIn("Richardson2020", citations._papers_to_cite)

        citations._reset()
        pybamm.ParameterValues("ORegan2021")
        self.assertIn("ORegan2021", citations._papers_to_cite)

    def test_solver_citations(self):
        # Test that solving each solver adds the right citations
        citations = pybamm.citations

        citations._reset()
        self.assertNotIn("Virtanen2020", citations._papers_to_cite)
        pybamm.ScipySolver()
        self.assertIn("Virtanen2020", citations._papers_to_cite)

        citations._reset()
        self.assertNotIn("Virtanen2020", citations._papers_to_cite)
        pybamm.AlgebraicSolver()
        self.assertIn("Virtanen2020", citations._papers_to_cite)

        if pybamm.have_scikits_odes():
            citations._reset()
            self.assertNotIn("Malengier2018", citations._papers_to_cite)
            pybamm.ScikitsOdeSolver()
            self.assertIn("Malengier2018", citations._papers_to_cite)

            citations._reset()
            self.assertNotIn("Malengier2018", citations._papers_to_cite)
            pybamm.ScikitsDaeSolver()
            self.assertIn("Malengier2018", citations._papers_to_cite)

        if pybamm.have_idaklu():
            citations._reset()
            self.assertNotIn("Hindmarsh2005", citations._papers_to_cite)
            pybamm.IDAKLUSolver()
            self.assertIn("Hindmarsh2005", citations._papers_to_cite)

    @unittest.skipIf(not pybamm.have_jax(), "jax or jaxlib is not installed")
    def test_jax_citations(self):
        citations = pybamm.citations
        citations._reset()
        self.assertNotIn("jax2018", citations._papers_to_cite)
        pybamm.JaxSolver()
        self.assertIn("jax2018", citations._papers_to_cite)


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
