#
# Tests the citations class.
#
import pybamm
import unittest


class TestCitations(unittest.TestCase):
    def test_citations(self):
        citations = pybamm.citations
        citations._reset()
        # Default papers should be in both _all_citations dict and in the papers to cite
        self.assertIn("sulzer2020python", citations._all_citations.keys())
        self.assertIn("sulzer2020python", citations._papers_to_cite)
        # Non-default papers should only be in the _all_citations dict
        self.assertIn("sulzer2019physical", citations._all_citations.keys())
        self.assertNotIn("sulzer2019physical", citations._papers_to_cite)

        # test key error
        with self.assertRaises(KeyError):
            citations.register("not a citation")

    def test_print_citations(self):
        pybamm.citations._reset()
        pybamm.print_citations("test_citations.txt")
        pybamm.citations._papers_to_cite = set()
        pybamm.print_citations()

    def test_Andersson_2019(self):
        citations = pybamm.citations
        citations._reset()
        self.assertNotIn("Andersson2019", citations._papers_to_cite)
        pybamm.CasadiConverter()
        self.assertIn("Andersson2019", citations._papers_to_cite)

    def test_marquis_2019(self):
        # Test that calling relevant bits of code adds the right paper to citations
        citations = pybamm.citations

        citations._reset()
        self.assertNotIn("marquis2019asymptotic", citations._papers_to_cite)
        pybamm.lithium_ion.SPM(build=False)
        self.assertIn("marquis2019asymptotic", citations._papers_to_cite)

        citations._reset()
        pybamm.lithium_ion.SPMe(build=False)
        self.assertIn("marquis2019asymptotic", citations._papers_to_cite)

    def test_doyle_1993(self):
        citations = pybamm.citations
        citations._reset()
        self.assertNotIn("doyle1993modeling", citations._papers_to_cite)
        pybamm.lithium_ion.DFN(build=False)
        self.assertIn("doyle1993modeling", citations._papers_to_cite)

    def test_sulzer_2019(self):
        # Test that calling relevant bits of code adds the right paper to citations
        citations = pybamm.citations

        citations._reset()
        self.assertNotIn("sulzer2019asymptotic", citations._papers_to_cite)
        pybamm.lead_acid.LOQS(build=False)
        self.assertIn("sulzer2019asymptotic", citations._papers_to_cite)

        citations._reset()
        pybamm.lead_acid.FOQS(build=False)
        self.assertIn("sulzer2019asymptotic", citations._papers_to_cite)

        citations._reset()
        pybamm.lead_acid.Composite(build=False)
        self.assertIn("sulzer2019asymptotic", citations._papers_to_cite)

        citations._reset()
        pybamm.lead_acid.Full(build=False)
        self.assertIn("sulzer2019physical", citations._papers_to_cite)

    def test_timms_2020(self):
        # Test that calling relevant bits of code adds the right paper to citations
        citations = pybamm.citations

        citations._reset()
        self.assertNotIn("timms2020", citations._papers_to_cite)
        pybamm.current_collector.BasePotentialPair(param=None)
        self.assertIn("timms2020", citations._papers_to_cite)

        citations._reset()
        self.assertNotIn("timms2020", citations._papers_to_cite)
        pybamm.current_collector.EffectiveResistance()
        self.assertIn("timms2020", citations._papers_to_cite)

        citations._reset()
        self.assertNotIn("timms2020", citations._papers_to_cite)
        pybamm.current_collector.AlternativeEffectiveResistance2D()
        self.assertIn("timms2020", citations._papers_to_cite)

    def test_subramanian_2005(self):
        # Test that calling relevant bits of code adds the right paper to citations
        citations = pybamm.citations

        citations._reset()
        self.assertNotIn("subramanian2005", citations._papers_to_cite)
        pybamm.particle.PolynomialSingleParticle(None, "Negative", "quadratic profile")
        self.assertIn("subramanian2005", citations._papers_to_cite)

        citations._reset()
        self.assertNotIn("subramanian2005", citations._papers_to_cite)
        pybamm.particle.PolynomialManyParticles(None, "Negative", "quadratic profile")
        self.assertIn("subramanian2005", citations._papers_to_cite)

    def test_scikit_fem(self):
        citations = pybamm.citations

        citations._reset()
        self.assertNotIn("scikit-fem", citations._papers_to_cite)
        pybamm.ScikitFiniteElement()
        self.assertIn("scikit-fem", citations._papers_to_cite)

    def test_parameter_citations(self):
        citations = pybamm.citations

        citations._reset()
        pybamm.ParameterValues(chemistry=pybamm.parameter_sets.Chen2020)
        self.assertIn("Chen2020", citations._papers_to_cite)

        citations._reset()
        pybamm.ParameterValues(chemistry=pybamm.parameter_sets.NCA_Kim2011)
        self.assertIn("kim2011multi", citations._papers_to_cite)

        citations._reset()
        pybamm.ParameterValues(chemistry=pybamm.parameter_sets.Marquis2019)
        self.assertIn("marquis2019asymptotic", citations._papers_to_cite)

        citations._reset()
        pybamm.ParameterValues(chemistry=pybamm.parameter_sets.Sulzer2019)
        self.assertIn("sulzer2019physical", citations._papers_to_cite)

        citations._reset()
        pybamm.ParameterValues(chemistry=pybamm.parameter_sets.Ecker2015)
        self.assertIn("ecker2015i", citations._papers_to_cite)
        self.assertIn("ecker2015ii", citations._papers_to_cite)
        self.assertIn("richardson2020", citations._papers_to_cite)

    def test_solver_citations(self):
        # Test that solving each solver adds the right citations
        citations = pybamm.citations

        citations._reset()
        self.assertNotIn("virtanen2020scipy", citations._papers_to_cite)
        pybamm.ScipySolver()
        self.assertIn("virtanen2020scipy", citations._papers_to_cite)

        citations._reset()
        self.assertNotIn("virtanen2020scipy", citations._papers_to_cite)
        pybamm.AlgebraicSolver()
        self.assertIn("virtanen2020scipy", citations._papers_to_cite)

        if pybamm.have_scikits_odes():
            citations._reset()
            self.assertNotIn("scikits-odes", citations._papers_to_cite)
            pybamm.ScikitsOdeSolver()
            self.assertIn("scikits-odes", citations._papers_to_cite)

            citations._reset()
            self.assertNotIn("scikits-odes", citations._papers_to_cite)
            pybamm.ScikitsDaeSolver()
            self.assertIn("scikits-odes", citations._papers_to_cite)

        if pybamm.have_idaklu():
            citations._reset()
            self.assertNotIn("hindmarsh2005sundials", citations._papers_to_cite)
            pybamm.IDAKLUSolver()
            self.assertIn("hindmarsh2005sundials", citations._papers_to_cite)


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
