#
# Tests the citations class.
#
import pybamm
import os
import io
import unittest
import contextlib
import warnings
from pybtex.database import Entry
from tempfile import NamedTemporaryFile


@contextlib.contextmanager
def temporary_filename():
    """Create a temporary-file and return yield its filename"""

    f = NamedTemporaryFile(delete=False)
    try:
        f.close()
        yield f.name
    finally:
        os.remove(f.name)


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

        # Register a citation that does not exist
        citations.register("not a citation")

        # Test key error
        with self.assertRaises(KeyError):
            citations._parse_citation("not a citation")  # this should raise key error

        # Test unknown citations at registration
        self.assertIn("not a citation", citations._unknown_citations)

    def test_print_citations(self):
        pybamm.citations._reset()

        # Text Style
        with temporary_filename() as filename:
            pybamm.print_citations(filename, "text")
            with open(filename) as f:
                self.assertTrue(len(f.readlines()) > 0)

        # Bibtext Style
        with temporary_filename() as filename:
            pybamm.print_citations(filename, "bibtex")
            with open(filename) as f:
                self.assertTrue(len(f.readlines()) > 0)

        # Write to stdout
        f = io.StringIO()
        with contextlib.redirect_stdout(f):
            pybamm.print_citations()
        self.assertTrue(
            "Python Battery Mathematical Modelling (PyBaMM)." in f.getvalue()
        )

        with self.assertRaisesRegex(pybamm.OptionError, "'text' or 'bibtex'"):
            pybamm.print_citations("test_citations.txt", "bad format")

        pybamm.citations._citation_err_msg = "Error"
        with self.assertRaisesRegex(ImportError, "Error"):
            pybamm.print_citations()
        pybamm.citations._citation_err_msg = None

        # Test that unknown citation raises warning message on printing
        pybamm.citations._reset()
        pybamm.citations.register("not a citation")
        with self.assertWarnsRegex(UserWarning, "not a citation"):
            pybamm.print_citations()

    def test_overwrite_citation(self):
        # Unknown citation
        fake_citation = r"@article{NotACitation, title = {This Doesn't Exist}}"
        with warnings.catch_warnings():
            pybamm.citations.register(fake_citation)
            pybamm.citations._parse_citation(fake_citation)
        self.assertIn("NotACitation", pybamm.citations._papers_to_cite)

        # Same NotACitation
        with warnings.catch_warnings():
            pybamm.citations.register(fake_citation)
            pybamm.citations._parse_citation(fake_citation)
        self.assertIn("NotACitation", pybamm.citations._papers_to_cite)

        # Overwrite NotACitation
        old_citation = pybamm.citations._all_citations["NotACitation"]
        with self.assertWarns(Warning):
            pybamm.citations.register(r"@article{NotACitation, title = {A New Title}}")
            pybamm.citations._parse_citation(
                r"@article{NotACitation, title = {A New Title}}"
            )
        self.assertIn("NotACitation", pybamm.citations._papers_to_cite)
        self.assertNotEqual(
            pybamm.citations._all_citations["NotACitation"], old_citation
        )

    def test_input_validation(self):
        """Test type validation of ``_add_citation``"""
        pybamm.citations.register(1)

        with self.assertRaises(TypeError):
            pybamm.citations._parse_citation(1)

        with self.assertRaises(TypeError):
            pybamm.citations._add_citation("NotACitation", "NotAEntry")

        with self.assertRaises(TypeError):
            pybamm.citations._add_citation(1001, Entry("misc"))

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
        self.assertIn("Marquis2019", citations._citation_tags.keys())

        citations._reset()
        pybamm.lithium_ion.SPMe(build=False)
        self.assertIn("Marquis2019", citations._papers_to_cite)

    def test_doyle_1993(self):
        citations = pybamm.citations
        citations._reset()
        self.assertNotIn("Doyle1993", citations._papers_to_cite)
        pybamm.lithium_ion.DFN(build=False)
        self.assertIn("Doyle1993", citations._papers_to_cite)
        self.assertIn("Doyle1993", citations._citation_tags.keys())

    def test_sulzer_2019(self):
        # Test that calling relevant bits of code adds the right paper to citations
        citations = pybamm.citations

        citations._reset()
        self.assertNotIn("Sulzer2019asymptotic", citations._papers_to_cite)
        pybamm.lead_acid.LOQS(build=False)
        self.assertIn("Sulzer2019asymptotic", citations._papers_to_cite)
        self.assertIn("Sulzer2019asymptotic", citations._citation_tags.keys())

        citations._reset()
        pybamm.lead_acid.Full(build=False)
        self.assertIn("Sulzer2019physical", citations._papers_to_cite)
        self.assertIn("Sulzer2019physical", citations._citation_tags.keys())

    def test_timms_2021(self):
        # Test that calling relevant bits of code adds the right paper to citations
        citations = pybamm.citations

        citations._reset()
        self.assertNotIn("Timms2021", citations._papers_to_cite)
        pybamm.current_collector.BasePotentialPair(param=None)
        self.assertIn("Timms2021", citations._papers_to_cite)
        self.assertIn("Timms2021", citations._citation_tags.keys())

        citations._reset()
        self.assertNotIn("Timms2021", citations._papers_to_cite)
        pybamm.current_collector.EffectiveResistance()
        self.assertIn("Timms2021", citations._papers_to_cite)
        self.assertIn("Timms2021", citations._citation_tags.keys())

        citations._reset()
        self.assertNotIn("Timms2021", citations._papers_to_cite)
        pybamm.current_collector.AlternativeEffectiveResistance2D()
        self.assertIn("Timms2021", citations._papers_to_cite)
        self.assertIn("Timms2021", citations._citation_tags.keys())

        citations._reset()
        self.assertNotIn("Timms2021", citations._papers_to_cite)
        pybamm.thermal.pouch_cell.CurrentCollector1D(param=None)
        self.assertIn("Timms2021", citations._papers_to_cite)
        self.assertIn("Timms2021", citations._citation_tags.keys())

        citations._reset()
        self.assertNotIn("Timms2021", citations._papers_to_cite)
        pybamm.thermal.pouch_cell.CurrentCollector2D(param=None)
        self.assertIn("Timms2021", citations._papers_to_cite)
        self.assertIn("Timms2021", citations._citation_tags.keys())

        citations._reset()
        self.assertNotIn("Timms2021", citations._papers_to_cite)
        pybamm.thermal.Lumped(param=None)
        self.assertIn("Timms2021", citations._papers_to_cite)
        self.assertIn("Timms2021", citations._citation_tags.keys())

        citations._reset()
        self.assertNotIn("Timms2021", citations._papers_to_cite)
        pybamm.thermal.pouch_cell.OneDimensionalX(param=None)
        self.assertIn("Timms2021", citations._papers_to_cite)
        self.assertIn("Timms2021", citations._citation_tags.keys())

    def test_subramanian_2005(self):
        # Test that calling relevant bits of code adds the right paper to citations
        citations = pybamm.citations

        citations._reset()
        self.assertNotIn("Subramanian2005", citations._papers_to_cite)
        pybamm.particle.XAveragedPolynomialProfile(
            None, "negative", {"particle": "quadratic profile"}, "primary"
        )
        self.assertIn("Subramanian2005", citations._papers_to_cite)
        self.assertIn("Subramanian2005", citations._citation_tags.keys())

        citations._reset()
        self.assertNotIn("Subramanian2005", citations._papers_to_cite)
        pybamm.particle.PolynomialProfile(
            None, "negative", {"particle": "quadratic profile"}, "primary"
        )
        self.assertIn("Subramanian2005", citations._papers_to_cite)
        self.assertIn("Subramanian2005", citations._citation_tags.keys())

    def test_brosaplanella_2021(self):
        # Test that calling relevant bits of code adds the right paper to citations
        citations = pybamm.citations

        citations._reset()
        self.assertNotIn("BrosaPlanella2021", citations._papers_to_cite)
        pybamm.electrolyte_conductivity.Integrated(None)
        self.assertIn("BrosaPlanella2021", citations._papers_to_cite)
        self.assertIn("BrosaPlanella2021", citations._citation_tags.keys())

    def test_brosaplanella_2022(self):
        # Test that calling relevant bits of code adds the right paper to citations
        citations = pybamm.citations

        citations._reset()
        self.assertNotIn("BrosaPlanella2022", citations._papers_to_cite)
        pybamm.lithium_ion.SPM(build=False, options={"SEI": "none"})
        pybamm.lithium_ion.SPM(build=False, options={"SEI": "constant"})
        pybamm.lithium_ion.SPMe(build=False, options={"SEI": "none"})
        pybamm.lithium_ion.SPMe(build=False, options={"SEI": "constant"})
        self.assertNotIn("BrosaPlanella2022", citations._papers_to_cite)

        pybamm.lithium_ion.SPM(build=False, options={"SEI": "ec reaction limited"})
        self.assertIn("BrosaPlanella2022", citations._papers_to_cite)
        citations._reset()

        pybamm.lithium_ion.SPMe(build=False, options={"SEI": "ec reaction limited"})
        self.assertIn("BrosaPlanella2022", citations._papers_to_cite)
        citations._reset()

        pybamm.lithium_ion.SPM(build=False, options={"lithium plating": "irreversible"})
        self.assertIn("BrosaPlanella2022", citations._papers_to_cite)
        citations._reset()

        pybamm.lithium_ion.SPMe(
            build=False, options={"lithium plating": "irreversible"}
        )
        self.assertIn("BrosaPlanella2022", citations._papers_to_cite)
        self.assertIn("BrosaPlanella2022", citations._citation_tags.keys())
        citations._reset()

    def test_newman_tobias(self):
        # Test that calling relevant bits of code adds the right paper to citations
        citations = pybamm.citations

        citations._reset()
        self.assertNotIn("Newman1962", citations._papers_to_cite)
        self.assertNotIn("Chu2020", citations._papers_to_cite)
        pybamm.lithium_ion.NewmanTobias()
        self.assertIn("Newman1962", citations._papers_to_cite)
        self.assertIn("Newman1962", citations._citation_tags.keys())
        self.assertIn("Chu2020", citations._papers_to_cite)
        self.assertIn("Chu2020", citations._citation_tags.keys())

    def test_scikit_fem(self):
        citations = pybamm.citations

        citations._reset()
        self.assertNotIn("Gustafsson2020", citations._papers_to_cite)
        pybamm.ScikitFiniteElement()
        self.assertIn("Gustafsson2020", citations._papers_to_cite)
        self.assertIn("Gustafsson2020", citations._citation_tags.keys())

    def test_reniers_2019(self):
        # Test that calling relevant bits of code adds the right paper to citations
        citations = pybamm.citations

        citations._reset()
        self.assertNotIn("Reniers2019", citations._papers_to_cite)
        pybamm.active_material.LossActiveMaterial(None, "negative", None, True)
        self.assertIn("Reniers2019", citations._papers_to_cite)
        self.assertIn("Reniers2019", citations._citation_tags.keys())

    def test_mohtat_2019(self):
        citations = pybamm.citations

        citations._reset()
        self.assertNotIn("Mohtat2019", citations._papers_to_cite)
        pybamm.lithium_ion.ElectrodeSOHSolver(
            pybamm.ParameterValues("Marquis2019")
        )._get_electrode_soh_sims_full()
        self.assertIn("Mohtat2019", citations._papers_to_cite)
        self.assertIn("Mohtat2019", citations._citation_tags.keys())

    def test_mohtat_2021(self):
        citations = pybamm.citations

        citations._reset()
        self.assertNotIn("Mohtat2021", citations._papers_to_cite)
        pybamm.external_circuit.CCCVFunctionControl(None, None)
        self.assertIn("Mohtat2021", citations._papers_to_cite)
        self.assertIn("Mohtat2021", citations._citation_tags.keys())

    def test_sripad_2020(self):
        citations = pybamm.citations

        citations._reset()
        self.assertNotIn("Sripad2020", citations._papers_to_cite)
        pybamm.kinetics.Marcus(None, None, None, None, None)
        self.assertIn("Sripad2020", citations._papers_to_cite)
        self.assertIn("Sripad2020", citations._citation_tags.keys())

        citations._reset()
        self.assertNotIn("Sripad2020", citations._papers_to_cite)
        pybamm.kinetics.MarcusHushChidsey(None, None, None, None, None)
        self.assertIn("Sripad2020", citations._papers_to_cite)
        self.assertIn("Sripad2020", citations._citation_tags.keys())

    def test_msmr(self):
        citations = pybamm.citations

        citations._reset()
        self.assertNotIn("Baker2018", citations._papers_to_cite)
        self.assertNotIn("Verbrugge2017", citations._papers_to_cite)
        pybamm.particle.MSMRDiffusion(None, "negative", None, None, None)
        self.assertIn("Baker2018", citations._papers_to_cite)
        self.assertIn("Baker2018", citations._citation_tags.keys())
        self.assertIn("Verbrugge2017", citations._papers_to_cite)
        self.assertIn("Verbrugge2017", citations._citation_tags.keys())

    def test_parameter_citations(self):
        citations = pybamm.citations

        citations._reset()
        pybamm.ParameterValues("Chen2020")
        self.assertIn("Chen2020", citations._papers_to_cite)
        self.assertIn("Chen2020", citations._citation_tags.keys())

        citations._reset()
        pybamm.ParameterValues("NCA_Kim2011")
        self.assertIn("Kim2011", citations._papers_to_cite)
        self.assertIn("Kim2011", citations._citation_tags.keys())

        citations._reset()
        pybamm.ParameterValues("Marquis2019")
        self.assertIn("Marquis2019", citations._papers_to_cite)
        self.assertIn("Marquis2019", citations._citation_tags.keys())

        citations._reset()
        pybamm.ParameterValues("Sulzer2019")
        self.assertIn("Sulzer2019physical", citations._papers_to_cite)
        self.assertIn("Sulzer2019physical", citations._citation_tags.keys())

        citations._reset()
        pybamm.ParameterValues("Ecker2015")
        self.assertIn("Ecker2015i", citations._papers_to_cite)
        self.assertIn("Ecker2015i", citations._citation_tags.keys())
        self.assertIn("Ecker2015ii", citations._papers_to_cite)
        self.assertIn("Ecker2015ii", citations._citation_tags.keys())
        self.assertIn("Zhao2018", citations._papers_to_cite)
        self.assertIn("Zhao2018", citations._citation_tags.keys())
        self.assertIn("Hales2019", citations._papers_to_cite)
        self.assertIn("Hales2019", citations._citation_tags.keys())
        self.assertIn("Richardson2020", citations._papers_to_cite)
        self.assertIn("Richardson2020", citations._citation_tags.keys())

        citations._reset()
        pybamm.ParameterValues("ORegan2022")
        self.assertIn("ORegan2022", citations._papers_to_cite)
        self.assertIn("ORegan2022", citations._citation_tags.keys())

        citations._reset()
        pybamm.ParameterValues("MSMR_Example")
        self.assertIn("Baker2018", citations._papers_to_cite)
        self.assertIn("Baker2018", citations._citation_tags.keys())
        self.assertIn("Verbrugge2017", citations._papers_to_cite)
        self.assertIn("Verbrugge2017", citations._citation_tags.keys())

    def test_solver_citations(self):
        # Test that solving each solver adds the right citations
        citations = pybamm.citations

        citations._reset()
        self.assertNotIn("Virtanen2020", citations._papers_to_cite)
        pybamm.ScipySolver()
        self.assertIn("Virtanen2020", citations._papers_to_cite)
        self.assertIn("Virtanen2020", citations._citation_tags.keys())

        citations._reset()
        self.assertNotIn("Virtanen2020", citations._papers_to_cite)
        pybamm.AlgebraicSolver()
        self.assertIn("Virtanen2020", citations._papers_to_cite)
        self.assertIn("Virtanen2020", citations._citation_tags.keys())

        if pybamm.have_scikits_odes():
            citations._reset()
            self.assertNotIn("Malengier2018", citations._papers_to_cite)
            pybamm.ScikitsOdeSolver()
            self.assertIn("Malengier2018", citations._papers_to_cite)
            self.assertIn("Malengier2018", citations._citation_tags.keys())

            citations._reset()
            self.assertNotIn("Malengier2018", citations._papers_to_cite)
            pybamm.ScikitsDaeSolver()
            self.assertIn("Malengier2018", citations._papers_to_cite)
            self.assertIn("Malengier2018", citations._citation_tags.keys())

        if pybamm.have_idaklu():
            citations._reset()
            self.assertNotIn("Hindmarsh2005", citations._papers_to_cite)
            pybamm.IDAKLUSolver()
            self.assertIn("Hindmarsh2005", citations._papers_to_cite)
            self.assertIn("Hindmarsh2005", citations._citation_tags.keys())

    @unittest.skipIf(not pybamm.have_jax(), "jax or jaxlib is not installed")
    def test_jax_citations(self):
        citations = pybamm.citations
        citations._reset()
        self.assertNotIn("jax2018", citations._papers_to_cite)
        pybamm.JaxSolver()
        self.assertIn("jax2018", citations._papers_to_cite)
        self.assertIn("jax2018", citations._citation_tags.keys())


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
