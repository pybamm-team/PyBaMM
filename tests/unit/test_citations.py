import pytest
import pybamm
import os
import io
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


class TestCitations:
    def test_citations(self):
        citations = pybamm.citations
        # Default papers should be in both _all_citations dict and in the papers to cite
        assert "Sulzer2021" in citations._all_citations.keys()
        assert "Sulzer2021" in citations._papers_to_cite
        assert "Harris2020" in citations._papers_to_cite
        # Non-default papers should only be in the _all_citations dict
        assert "Sulzer2019physical" in citations._all_citations.keys()
        assert "Sulzer2019physical" not in citations._papers_to_cite

        # Register a citation that does not exist
        citations.register("not a citation")

        # Test key error
        with pytest.raises(KeyError):
            citations._parse_citation("not a citation")  # this should raise key error

        # Test unknown citations at registration
        assert "not a citation" in citations._unknown_citations

    def test_print_citations(self):
        pybamm.citations._reset()

        # Text Style
        with temporary_filename() as filename:
            pybamm.print_citations(filename, "text")
            with open(filename) as f:
                assert len(f.readlines()) > 0

        # Bibtext Style
        with temporary_filename() as filename:
            pybamm.print_citations(filename, "bibtex")
            with open(filename) as f:
                assert len(f.readlines()) > 0

        # Write to stdout
        f = io.StringIO()
        with contextlib.redirect_stdout(f):
            pybamm.print_citations()
        assert "Python Battery Mathematical Modelling (PyBaMM)." in f.getvalue()

        with pytest.raises(pybamm.OptionError, match="'text' or 'bibtex'"):
            pybamm.print_citations("test_citations.txt", "bad format")

        # Test that unknown citation raises warning message on printing
        pybamm.citations._reset()
        pybamm.citations.register("not a citation")
        with pytest.warns(UserWarning, match="not a citation"):
            pybamm.print_citations()

    def test_overwrite_citation(self):
        # Unknown citation
        fake_citation = r"@article{NotACitation, title = {This Doesn't Exist}}"
        with warnings.catch_warnings():
            pybamm.citations.register(fake_citation)
            pybamm.citations._parse_citation(fake_citation)
        assert "NotACitation" in pybamm.citations._papers_to_cite

        # Same NotACitation
        with warnings.catch_warnings():
            pybamm.citations.register(fake_citation)
            pybamm.citations._parse_citation(fake_citation)
        assert "NotACitation" in pybamm.citations._papers_to_cite

        # Overwrite NotACitation
        old_citation = pybamm.citations._all_citations["NotACitation"]
        with pytest.warns(Warning):
            pybamm.citations.register(r"@article{NotACitation, title = {A New Title}}")
            pybamm.citations._parse_citation(
                r"@article{NotACitation, title = {A New Title}}"
            )
        assert "NotACitation" in pybamm.citations._papers_to_cite
        assert pybamm.citations._all_citations["NotACitation"] != old_citation

    def test_input_validation(self):
        """Test type validation of ``_add_citation``"""
        pybamm.citations.register(1)

        with pytest.raises(TypeError):
            pybamm.citations._parse_citation(1)

        with pytest.raises(TypeError):
            pybamm.citations._add_citation("NotACitation", "NotAEntry")

        with pytest.raises(TypeError):
            pybamm.citations._add_citation(1001, Entry("misc"))

    def test_pybtex_warning(self, caplog):
        class CiteWithWarning(pybamm.Citations):
            def __init__(self):
                super().__init__()
                self._module_import_error = True

        CiteWithWarning().print_import_warning()
        assert "Could not print citations" in caplog.text

    def test_andersson_2019(self):
        citations = pybamm.citations
        citations._reset()
        assert "Andersson2019" not in citations._papers_to_cite
        pybamm.CasadiConverter()
        assert "Andersson2019" in citations._papers_to_cite

    def test_marquis_2019(self):
        # Test that calling relevant bits of code adds the right paper to citations
        citations = pybamm.citations

        citations._reset()
        assert "Marquis2019" not in citations._papers_to_cite
        pybamm.lithium_ion.SPM(build=False)
        assert "Marquis2019" in citations._papers_to_cite
        assert "Marquis2019" in citations._citation_tags.keys()

        citations._reset()
        pybamm.lithium_ion.SPMe(build=False)
        assert "Marquis2019" in citations._papers_to_cite

    def test_doyle_1993(self):
        citations = pybamm.citations
        citations._reset()
        assert "Doyle1993" not in citations._papers_to_cite
        pybamm.lithium_ion.DFN(build=False)
        assert "Doyle1993" in citations._papers_to_cite
        assert "Doyle1993" in citations._citation_tags.keys()

    def test_sulzer_2019(self):
        # Test that calling relevant bits of code adds the right paper to citations
        citations = pybamm.citations

        citations._reset()
        assert "Sulzer2019asymptotic" not in citations._papers_to_cite
        pybamm.lead_acid.LOQS(build=False)
        assert "Sulzer2019asymptotic" in citations._papers_to_cite
        assert "Sulzer2019asymptotic" in citations._citation_tags.keys()

        citations._reset()
        pybamm.lead_acid.Full(build=False)
        assert "Sulzer2019physical" in citations._papers_to_cite
        assert "Sulzer2019physical" in citations._citation_tags.keys()

    def test_timms_2021(self):
        # Test that calling relevant bits of code adds the right paper to citations
        citations = pybamm.citations

        citations._reset()
        assert "Timms2021" not in citations._papers_to_cite
        pybamm.current_collector.BasePotentialPair(param=None)
        assert "Timms2021" in citations._papers_to_cite
        assert "Timms2021" in citations._citation_tags.keys()

        citations._reset()
        assert "Timms2021" not in citations._papers_to_cite
        pybamm.current_collector.EffectiveResistance()
        assert "Timms2021" in citations._papers_to_cite
        assert "Timms2021" in citations._citation_tags.keys()

        citations._reset()
        assert "Timms2021" not in citations._papers_to_cite
        pybamm.current_collector.AlternativeEffectiveResistance2D()
        assert "Timms2021" in citations._papers_to_cite
        assert "Timms2021" in citations._citation_tags.keys()

        citations._reset()
        assert "Timms2021" not in citations._papers_to_cite
        pybamm.thermal.pouch_cell.CurrentCollector1D(param=None)
        assert "Timms2021" in citations._papers_to_cite
        assert "Timms2021" in citations._citation_tags.keys()

        citations._reset()
        assert "Timms2021" not in citations._papers_to_cite
        pybamm.thermal.pouch_cell.CurrentCollector2D(param=None)
        assert "Timms2021" in citations._papers_to_cite
        assert "Timms2021" in citations._citation_tags.keys()

        citations._reset()
        assert "Timms2021" not in citations._papers_to_cite
        pybamm.thermal.Lumped(param=None)
        assert "Timms2021" in citations._papers_to_cite
        assert "Timms2021" in citations._citation_tags.keys()

        citations._reset()
        assert "Timms2021" not in citations._papers_to_cite
        pybamm.thermal.pouch_cell.OneDimensionalX(param=None)
        assert "Timms2021" in citations._papers_to_cite
        assert "Timms2021" in citations._citation_tags.keys()

    def test_subramanian_2005(self):
        # Test that calling relevant bits of code adds the right paper to citations
        citations = pybamm.citations

        citations._reset()
        assert "Subramanian2005" not in citations._papers_to_cite
        pybamm.particle.XAveragedPolynomialProfile(
            None, "negative", {"particle": "quadratic profile"}, "primary"
        )
        assert "Subramanian2005" in citations._papers_to_cite
        assert "Subramanian2005" in citations._citation_tags.keys()

        citations._reset()
        assert "Subramanian2005" not in citations._papers_to_cite
        pybamm.particle.PolynomialProfile(
            None, "negative", {"particle": "quadratic profile"}, "primary"
        )
        assert "Subramanian2005" in citations._papers_to_cite
        assert "Subramanian2005" in citations._citation_tags.keys()

    def test_brosaplanella_2021(self):
        # Test that calling relevant bits of code adds the right paper to citations
        citations = pybamm.citations

        citations._reset()
        assert "BrosaPlanella2021" not in citations._papers_to_cite
        pybamm.electrolyte_conductivity.Integrated(None)
        assert "BrosaPlanella2021" in citations._papers_to_cite
        assert "BrosaPlanella2021" in citations._citation_tags.keys()

    def test_brosaplanella_2022(self):
        # Test that calling relevant bits of code adds the right paper to citations
        citations = pybamm.citations

        citations._reset()
        assert "BrosaPlanella2022" not in citations._papers_to_cite
        pybamm.lithium_ion.SPM(build=False, options={"SEI": "none"})
        pybamm.lithium_ion.SPM(build=False, options={"SEI": "constant"})
        pybamm.lithium_ion.SPMe(build=False, options={"SEI": "none"})
        pybamm.lithium_ion.SPMe(build=False, options={"SEI": "constant"})
        assert "BrosaPlanella2022" not in citations._papers_to_cite

        pybamm.lithium_ion.SPM(build=False, options={"SEI": "ec reaction limited"})
        assert "BrosaPlanella2022" in citations._papers_to_cite
        citations._reset()

        pybamm.lithium_ion.SPMe(build=False, options={"SEI": "ec reaction limited"})
        assert "BrosaPlanella2022" in citations._papers_to_cite
        citations._reset()

        pybamm.lithium_ion.SPM(build=False, options={"lithium plating": "irreversible"})
        assert "BrosaPlanella2022" in citations._papers_to_cite
        citations._reset()

        pybamm.lithium_ion.SPMe(
            build=False, options={"lithium plating": "irreversible"}
        )
        assert "BrosaPlanella2022" in citations._papers_to_cite
        assert "BrosaPlanella2022" in citations._citation_tags.keys()
        citations._reset()

    def test_newman_tobias(self):
        # Test that calling relevant bits of code adds the right paper to citations
        citations = pybamm.citations

        citations._reset()
        assert "Newman1962" not in citations._papers_to_cite
        assert "Chu2020" not in citations._papers_to_cite
        pybamm.lithium_ion.NewmanTobias()
        assert "Newman1962" in citations._papers_to_cite
        assert "Newman1962" in citations._citation_tags.keys()
        assert "Chu2020" in citations._papers_to_cite
        assert "Chu2020" in citations._citation_tags.keys()

    def test_scikit_fem(self):
        citations = pybamm.citations

        citations._reset()
        assert "Gustafsson2020" not in citations._papers_to_cite
        pybamm.ScikitFiniteElement()
        assert "Gustafsson2020" in citations._papers_to_cite
        assert "Gustafsson2020" in citations._citation_tags.keys()

    def test_reniers_2019(self):
        # Test that calling relevant bits of code adds the right paper to citations
        citations = pybamm.citations

        citations._reset()
        assert "Reniers2019" not in citations._papers_to_cite
        pybamm.active_material.LossActiveMaterial(None, "negative", None, True)
        assert "Reniers2019" in citations._papers_to_cite
        assert "Reniers2019" in citations._citation_tags.keys()

    def test_mohtat_2019(self):
        citations = pybamm.citations

        citations._reset()
        assert "Mohtat2019" not in citations._papers_to_cite
        pybamm.lithium_ion.ElectrodeSOHSolver(
            pybamm.ParameterValues("Marquis2019")
        )._get_electrode_soh_sims_full()
        assert "Mohtat2019" in citations._papers_to_cite
        assert "Mohtat2019" in citations._citation_tags.keys()

    def test_mohtat_2021(self):
        citations = pybamm.citations

        citations._reset()
        assert "Mohtat2021" not in citations._papers_to_cite
        pybamm.external_circuit.CCCVFunctionControl(None, None)
        assert "Mohtat2021" in citations._papers_to_cite
        assert "Mohtat2021" in citations._citation_tags.keys()

    def test_sripad_2020(self):
        citations = pybamm.citations

        citations._reset()
        assert "Sripad2020" not in citations._papers_to_cite
        pybamm.kinetics.Marcus(None, None, None, None, None)
        assert "Sripad2020" in citations._papers_to_cite
        assert "Sripad2020" in citations._citation_tags.keys()

        citations._reset()
        assert "Sripad2020" not in citations._papers_to_cite
        pybamm.kinetics.MarcusHushChidsey(None, None, None, None, None)
        assert "Sripad2020" in citations._papers_to_cite
        assert "Sripad2020" in citations._citation_tags.keys()

    def test_msmr(self):
        citations = pybamm.citations

        citations._reset()
        assert "Baker2018" not in citations._papers_to_cite
        assert "Verbrugge2017" not in citations._papers_to_cite
        pybamm.particle.MSMRDiffusion(None, "negative", None, None, None)
        assert "Baker2018" in citations._papers_to_cite
        assert "Baker2018" in citations._citation_tags.keys()
        assert "Verbrugge2017" in citations._papers_to_cite
        assert "Verbrugge2017" in citations._citation_tags.keys()

    def test_thevenin(self):
        citations = pybamm.citations

        citations._reset()
        pybamm.equivalent_circuit.Thevenin()
        assert "Fan2022" not in citations._papers_to_cite
        assert "Fan2022" not in citations._citation_tags.keys()

        pybamm.equivalent_circuit.Thevenin(options={"diffusion element": "true"})
        assert "Fan2022" in citations._papers_to_cite
        assert "Fan2022" in citations._citation_tags.keys()

    def test_parameter_citations(self):
        citations = pybamm.citations

        citations._reset()
        pybamm.ParameterValues("Chen2020")
        assert "Chen2020" in citations._papers_to_cite
        assert "Chen2020" in citations._citation_tags.keys()

        citations._reset()
        pybamm.ParameterValues("NCA_Kim2011")
        assert "Kim2011" in citations._papers_to_cite
        assert "Kim2011" in citations._citation_tags.keys()

        citations._reset()
        pybamm.ParameterValues("Marquis2019")
        assert "Marquis2019" in citations._papers_to_cite
        assert "Marquis2019" in citations._citation_tags.keys()

        citations._reset()
        pybamm.ParameterValues("Sulzer2019")
        assert "Sulzer2019physical" in citations._papers_to_cite
        assert "Sulzer2019physical" in citations._citation_tags.keys()

        citations._reset()
        pybamm.ParameterValues("Ecker2015")
        assert "Ecker2015i" in citations._papers_to_cite
        assert "Ecker2015i" in citations._citation_tags.keys()
        assert "Ecker2015ii" in citations._papers_to_cite
        assert "Ecker2015ii" in citations._citation_tags.keys()
        assert "Zhao2018" in citations._papers_to_cite
        assert "Zhao2018" in citations._citation_tags.keys()
        assert "Hales2019" in citations._papers_to_cite
        assert "Hales2019" in citations._citation_tags.keys()
        assert "Richardson2020" in citations._papers_to_cite
        assert "Richardson2020" in citations._citation_tags.keys()

        citations._reset()
        pybamm.ParameterValues("ORegan2022")
        assert "ORegan2022" in citations._papers_to_cite
        assert "ORegan2022" in citations._citation_tags.keys()

        citations._reset()
        pybamm.ParameterValues("MSMR_Example")
        assert "Baker2018" in citations._papers_to_cite
        assert "Baker2018" in citations._citation_tags.keys()
        assert "Verbrugge2017" in citations._papers_to_cite
        assert "Verbrugge2017" in citations._citation_tags.keys()

    def test_solver_citations(self):
        # Test that solving each solver adds the right citations
        citations = pybamm.citations

        citations._reset()
        assert "Virtanen2020" not in citations._papers_to_cite
        pybamm.ScipySolver()
        assert "Virtanen2020" in citations._papers_to_cite
        assert "Virtanen2020" in citations._citation_tags.keys()

        citations._reset()
        assert "Virtanen2020" not in citations._papers_to_cite
        pybamm.AlgebraicSolver()
        assert "Virtanen2020" in citations._papers_to_cite
        assert "Virtanen2020" in citations._citation_tags.keys()

        if pybamm.have_idaklu():
            citations._reset()
            assert "Hindmarsh2005" not in citations._papers_to_cite
            pybamm.IDAKLUSolver()
            assert "Hindmarsh2005" in citations._papers_to_cite
            assert "Hindmarsh2005" in citations._citation_tags.keys()

    @pytest.mark.skipif(not pybamm.have_jax(), reason="jax or jaxlib is not installed")
    def test_jax_citations(self):
        citations = pybamm.citations
        citations._reset()
        assert "jax2018" not in citations._papers_to_cite
        pybamm.JaxSolver()
        assert "jax2018" in citations._papers_to_cite
        assert "jax2018" in citations._citation_tags.keys()
