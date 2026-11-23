import contextlib
import importlib.util
import io
import os
import warnings

import pytest
from pybtex.database import Entry, parse_file
from pybtex.exceptions import PybtexError

import pybamm


class TestCitations:
    @pytest.fixture
    def citation_database(self, monkeypatch):
        citations = pybamm.citations
        # Resetting registrations leaves the bibliography and its key index in place.
        monkeypatch.setattr(
            citations, "_all_citations", citations._all_citations.copy()
        )
        monkeypatch.setattr(
            citations, "_citation_keys", citations._citation_keys.copy()
        )
        yield
        citations._reset()

    def test_citations(self):
        citations = pybamm.citations
        # Default papers should be in both _all_citations dict and in the papers to cite
        assert "Sulzer2021" in citations._all_citations
        assert "Sulzer2021" in citations._papers_to_cite
        assert "Harris2020" in citations._papers_to_cite
        # Non-default papers should only be in the _all_citations dict
        assert "Sulzer2019physical" in citations._all_citations
        assert "Sulzer2019physical" not in citations._papers_to_cite

        # Register a citation that does not exist
        citations.register("not a citation")

        # Test key error
        with pytest.raises(KeyError):
            citations._parse_citation("not a citation")

        # Test unknown citations at registration
        assert "not a citation" in citations._unknown_citations

    def test_print_citations(self, tmp_path):
        pybamm.citations._reset()

        # Text Style
        text_file = tmp_path / "citations.txt"
        pybamm.print_citations(text_file, "text")
        assert text_file.read_text().strip() != ""

        # Bibtext Style
        bibtex_file = tmp_path / "citations.bib"
        pybamm.print_citations(bibtex_file, "bibtex")
        assert bibtex_file.read_text().strip() != ""

        # Write to stdout
        f = io.StringIO()
        with contextlib.redirect_stdout(f):
            pybamm.print_citations()
        assert "Python Battery Mathematical Modelling (PyBaMM)." in f.getvalue()

        with pytest.raises(pybamm.OptionError, match=r"'text' or 'bibtex'"):
            pybamm.print_citations("test_citations.txt", "bad format")

        # Test that unknown citation raises warning message on printing
        pybamm.citations._reset()
        pybamm.citations.register("not a citation")
        with pytest.warns(UserWarning, match=r"not a citation"):
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
        new_citation = r"@article{NotACitation, title = {A New Title}}"
        pybamm.citations.register(new_citation)
        # the overwrite warning comes from parsing, not from registering
        with pytest.warns(Warning):
            pybamm.citations._parse_citation(new_citation)
        assert "NotACitation" in pybamm.citations._papers_to_cite
        assert pybamm.citations._all_citations["NotACitation"] != old_citation

    def test_stored_citations_match_bibtex_format(self):
        citations = pybamm.Citations()
        bib_data = parse_file(
            os.path.join(pybamm.__path__[0], "CITATIONS.bib"), bib_format="bibtex"
        )
        assert bib_data.entries
        # Keys are read without pybtex, so both must find the same entries
        assert citations._citation_keys == set(bib_data.entries.keys())
        for key, entry in bib_data.entries.items():
            assert citations._all_citations[key] == entry.to_string("bibtex")

    def test_register_known_key_defers_parsing(self):
        citations = pybamm.Citations()
        citations.register("Marquis2019")
        assert "Marquis2019" in citations._papers_to_cite
        assert "_all_citations" not in vars(citations)

    def test_read_citations_rereads_the_bibtex_file(self):
        citations = pybamm.Citations()
        entries = dict(citations._all_citations)
        # The same entries again, so nothing is replaced (a warning would fail)
        citations.read_citations()
        assert citations._all_citations == entries
        assert citations._citation_keys == set(entries)

    def test_construct_without_pybtex(self, monkeypatch):
        find_spec = importlib.util.find_spec
        monkeypatch.setattr(
            importlib.util,
            "find_spec",
            lambda name, *args: None if name == "pybtex" else find_spec(name, *args),
        )
        citations = pybamm.Citations()
        assert citations._module_import_error
        assert citations._citation_keys == set()
        assert citations._all_citations == {}
        citations.read_citations()
        assert citations._all_citations == {}
        # Without the keys, a CITATIONS.bib key is held back as unknown
        citations.register("Marquis2019")
        assert "Marquis2019" in citations._unknown_citations
        assert "Marquis2019" not in citations._papers_to_cite

    def test_input_validation(self):
        """Test type validation of ``_add_citation``"""
        pybamm.citations.register(1)

        with pytest.raises(TypeError, match="reason"):
            pybamm.citations.register("Marquis2019", reason=123)

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
        pybamm.Scalar(1).to_casadi()
        assert "Andersson2019" in citations._papers_to_cite

    def test_marquis_2019(self):
        # Test that calling relevant bits of code adds the right paper to citations
        citations = pybamm.citations

        citations._reset()
        assert "Marquis2019" not in citations._papers_to_cite
        pybamm.lithium_ion.SPM(build=False)
        assert "Marquis2019" in citations._papers_to_cite
        assert "Marquis2019" in citations._citation_tags

        citations._reset()
        pybamm.lithium_ion.SPMe(build=False)
        assert "Marquis2019" in citations._papers_to_cite

    def test_doyle_1993(self):
        citations = pybamm.citations
        citations._reset()
        assert "Doyle1993" not in citations._papers_to_cite
        pybamm.lithium_ion.DFN(build=False)
        assert "Doyle1993" in citations._papers_to_cite
        assert "Doyle1993" in citations._citation_tags

    def test_sulzer_2019(self):
        # Test that calling relevant bits of code adds the right paper to citations
        citations = pybamm.citations

        citations._reset()
        assert "Sulzer2019asymptotic" not in citations._papers_to_cite
        pybamm.lead_acid.LOQS(build=False)
        assert "Sulzer2019asymptotic" in citations._papers_to_cite
        assert "Sulzer2019asymptotic" in citations._citation_tags

        citations._reset()
        pybamm.lead_acid.Full(build=False)
        assert "Sulzer2019physical" in citations._papers_to_cite
        assert "Sulzer2019physical" in citations._citation_tags

    def test_timms_2021(self):
        # Test that calling relevant bits of code adds the right paper to citations
        citations = pybamm.citations

        citations._reset()
        assert "Timms2021" not in citations._papers_to_cite
        pybamm.current_collector.BasePotentialPair(param=None)
        assert "Timms2021" in citations._papers_to_cite
        assert "Timms2021" in citations._citation_tags

        citations._reset()
        assert "Timms2021" not in citations._papers_to_cite
        pybamm.current_collector.EffectiveResistance()
        assert "Timms2021" in citations._papers_to_cite
        assert "Timms2021" in citations._citation_tags

        citations._reset()
        assert "Timms2021" not in citations._papers_to_cite
        pybamm.current_collector.AlternativeEffectiveResistance2D()
        assert "Timms2021" in citations._papers_to_cite
        assert "Timms2021" in citations._citation_tags

        citations._reset()
        assert "Timms2021" not in citations._papers_to_cite
        pybamm.thermal.pouch_cell.CurrentCollector1D(param=None)
        assert "Timms2021" in citations._papers_to_cite
        assert "Timms2021" in citations._citation_tags

        citations._reset()
        assert "Timms2021" not in citations._papers_to_cite
        pybamm.thermal.pouch_cell.CurrentCollector2D(param=None)
        assert "Timms2021" in citations._papers_to_cite
        assert "Timms2021" in citations._citation_tags

        citations._reset()
        assert "Timms2021" not in citations._papers_to_cite
        pybamm.thermal.Lumped(param=None)
        assert "Timms2021" in citations._papers_to_cite
        assert "Timms2021" in citations._citation_tags

        citations._reset()
        assert "Timms2021" not in citations._papers_to_cite
        pybamm.thermal.pouch_cell.OneDimensionalX(param=None)
        assert "Timms2021" in citations._papers_to_cite
        assert "Timms2021" in citations._citation_tags

    def test_subramanian_2005(self):
        # Test that calling relevant bits of code adds the right paper to citations
        citations = pybamm.citations

        citations._reset()
        assert "Subramanian2005" not in citations._papers_to_cite
        pybamm.particle.XAveragedPolynomialProfile(
            None, "negative", {"particle": "quadratic profile"}, "primary"
        )
        assert "Subramanian2005" in citations._papers_to_cite
        assert "Subramanian2005" in citations._citation_tags

        citations._reset()
        assert "Subramanian2005" not in citations._papers_to_cite
        pybamm.particle.PolynomialProfile(
            None, "negative", {"particle": "quadratic profile"}, "primary"
        )
        assert "Subramanian2005" in citations._papers_to_cite
        assert "Subramanian2005" in citations._citation_tags

    def test_brosaplanella_2021(self):
        # Test that calling relevant bits of code adds the right paper to citations
        citations = pybamm.citations

        citations._reset()
        assert "BrosaPlanella2021" not in citations._papers_to_cite
        pybamm.electrolyte_conductivity.Integrated(None)
        assert "BrosaPlanella2021" in citations._papers_to_cite
        assert "BrosaPlanella2021" in citations._citation_tags

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
        assert "BrosaPlanella2022" in citations._citation_tags
        citations._reset()

    def test_VonKolzenberg_2020(self):
        # Test that calling relevant bits of code adds the right paper to citations
        citations = pybamm.citations

        citations._reset()
        assert "VonKolzenberg2020" not in citations._papers_to_cite

        pybamm.lithium_ion.SPMe(build=False, options={"SEI": "VonKolzenberg2020"})
        assert "VonKolzenberg2020" in citations._papers_to_cite
        citations._reset()

        pybamm.lithium_ion.SPM(build=False, options={"SEI": "VonKolzenberg2020"})
        assert "VonKolzenberg2020" in citations._papers_to_cite
        citations._reset()

    def test_tang_2012(self):
        # Test that calling relevant bits of code adds the right paper to citations
        citations = pybamm.citations

        citations._reset()
        assert "Tang2012" not in citations._papers_to_cite

        pybamm.lithium_ion.SPMe(build=False, options={"SEI": "tunnelling limited"})
        assert "Tang2012" in citations._papers_to_cite
        citations._reset()

        pybamm.lithium_ion.SPM(build=False, options={"SEI": "tunnelling limited"})
        assert "Tang2012" in citations._papers_to_cite
        citations._reset()

    def test_newman_tobias(self):
        # Test that calling relevant bits of code adds the right paper to citations
        citations = pybamm.citations

        citations._reset()
        assert "Newman1962" not in citations._papers_to_cite
        assert "Chu2020" not in citations._papers_to_cite
        pybamm.lithium_ion.NewmanTobias()
        assert "Newman1962" in citations._papers_to_cite
        assert "Newman1962" in citations._citation_tags
        assert "Chu2020" in citations._papers_to_cite
        assert "Chu2020" in citations._citation_tags

    def test_scikit_fem(self):
        citations = pybamm.citations

        citations._reset()
        assert "Gustafsson2020" not in citations._papers_to_cite
        pybamm.ScikitFiniteElement()
        assert "Gustafsson2020" in citations._papers_to_cite
        assert "Gustafsson2020" in citations._citation_tags

    def test_reniers_2019(self):
        # Test that calling relevant bits of code adds the right paper to citations
        citations = pybamm.citations

        citations._reset()
        assert "Reniers2019" not in citations._papers_to_cite
        pybamm.active_material.LossActiveMaterial(None, "negative", None, True, None)
        assert "Reniers2019" in citations._papers_to_cite
        assert "Reniers2019" in citations._citation_tags

    def test_mohtat_2019(self):
        citations = pybamm.citations

        citations._reset()
        assert "Mohtat2019" not in citations._papers_to_cite
        pybamm.lithium_ion.ElectrodeSOHSolver(
            pybamm.ParameterValues("Marquis2019")
        )._get_electrode_soh_sims_full(None)
        assert "Mohtat2019" in citations._papers_to_cite
        assert "Mohtat2019" in citations._citation_tags

    def test_mohtat_2021(self):
        citations = pybamm.citations

        citations._reset()
        assert "Mohtat2021" not in citations._papers_to_cite
        pybamm.external_circuit.CCCVFunctionControl(None, None)
        assert "Mohtat2021" in citations._papers_to_cite
        assert "Mohtat2021" in citations._citation_tags

    def test_sripad_2020(self):
        citations = pybamm.citations

        citations._reset()
        assert "Sripad2020" not in citations._papers_to_cite
        pybamm.kinetics.Marcus(None, "negative", None, None, None)
        assert "Sripad2020" in citations._papers_to_cite
        assert "Sripad2020" in citations._citation_tags

        citations._reset()
        assert "Sripad2020" not in citations._papers_to_cite
        pybamm.kinetics.MarcusHushChidsey(None, "negative", None, None, None)
        assert "Sripad2020" in citations._papers_to_cite
        assert "Sripad2020" in citations._citation_tags

    def test_msmr(self):
        citations = pybamm.citations

        citations._reset()
        assert "Baker2018" not in citations._papers_to_cite
        assert "Verbrugge2017" not in citations._papers_to_cite
        pybamm.particle.MSMRDiffusion(None, "negative", None, None, None)
        assert "Baker2018" in citations._papers_to_cite
        assert "Baker2018" in citations._citation_tags
        assert "Verbrugge2017" in citations._papers_to_cite
        assert "Verbrugge2017" in citations._citation_tags

    def test_thevenin(self):
        citations = pybamm.citations

        citations._reset()
        pybamm.equivalent_circuit.Thevenin()
        assert "Fan2022" not in citations._papers_to_cite
        assert "Fan2022" not in citations._citation_tags

        pybamm.equivalent_circuit.Thevenin(options={"diffusion element": "true"})
        assert "Fan2022" in citations._papers_to_cite
        assert "Fan2022" in citations._citation_tags

    def test_parameter_citations(self):
        citations = pybamm.citations

        citations._reset()
        pybamm.ParameterValues("Chen2020")
        assert "Chen2020" in citations._papers_to_cite
        assert "Chen2020" in citations._citation_tags

        citations._reset()
        pybamm.ParameterValues("NCA_Kim2011")
        assert "Kim2011" in citations._papers_to_cite
        assert "Kim2011" in citations._citation_tags

        citations._reset()
        pybamm.ParameterValues("Marquis2019")
        assert "Marquis2019" in citations._papers_to_cite
        assert "Marquis2019" in citations._citation_tags

        citations._reset()
        pybamm.ParameterValues("Sulzer2019")
        assert "Sulzer2019physical" in citations._papers_to_cite
        assert "Sulzer2019physical" in citations._citation_tags

        citations._reset()
        pybamm.ParameterValues("Ecker2015")
        assert "Ecker2015i" in citations._papers_to_cite
        assert "Ecker2015i" in citations._citation_tags
        assert "Ecker2015ii" in citations._papers_to_cite
        assert "Ecker2015ii" in citations._citation_tags
        assert "Zhao2018" in citations._papers_to_cite
        assert "Zhao2018" in citations._citation_tags
        assert "Hales2019" in citations._papers_to_cite
        assert "Hales2019" in citations._citation_tags
        assert "Richardson2020" in citations._papers_to_cite
        assert "Richardson2020" in citations._citation_tags

        citations._reset()
        pybamm.ParameterValues("ORegan2022")
        assert "ORegan2022" in citations._papers_to_cite
        assert "ORegan2022" in citations._citation_tags

        citations._reset()
        pybamm.ParameterValues("MSMR_Example")
        assert "Baker2018" in citations._papers_to_cite
        assert "Baker2018" in citations._citation_tags
        assert "Verbrugge2017" in citations._papers_to_cite
        assert "Verbrugge2017" in citations._citation_tags

    def test_solver_citations(self):
        # Test that solving each solver adds the right citations
        citations = pybamm.citations

        citations._reset()
        assert "Virtanen2020" not in citations._papers_to_cite
        pybamm.ScipySolver()
        assert "Virtanen2020" in citations._papers_to_cite
        assert "Virtanen2020" in citations._citation_tags

        citations._reset()
        assert "Virtanen2020" not in citations._papers_to_cite
        pybamm.AlgebraicSolver()
        assert "Virtanen2020" in citations._papers_to_cite
        assert "Virtanen2020" in citations._citation_tags

        citations._reset()
        assert "Hindmarsh2005" not in citations._papers_to_cite
        pybamm.IDAKLUSolver()
        assert "Hindmarsh2005" in citations._papers_to_cite
        assert "Hindmarsh2005" in citations._citation_tags

    @pytest.mark.skipif(not pybamm.has_jax(), reason="jax or jaxlib is not installed")
    def test_jax_citations(self):
        citations = pybamm.citations
        citations._reset()
        assert "jax2018" not in citations._papers_to_cite
        pybamm.JaxSolver()
        assert "jax2018" in citations._papers_to_cite
        assert "jax2018" in citations._citation_tags

    def test_default_reasons(self):
        citations = pybamm.citations
        citations._reset()
        assert citations._citation_tags == {
            "Sulzer2021": {"PyBaMM"},
            "Harris2020": {"NumPy"},
        }

    @pytest.mark.parametrize("as_string", [False, True])
    def test_external_database_does_not_cite(
        self, citation_database, tmp_path, as_string
    ):
        citations = pybamm.citations
        citations._reset()
        path = tmp_path / "downstream.bib"
        path.write_text("@misc{Downstream2026, title={Downstream example}}")
        registered = citations._papers_to_cite.copy()

        citations.read_citations(str(path) if as_string else path)

        assert "Downstream2026" in citations._all_citations
        assert citations._papers_to_cite == registered

    def test_external_database_missing_file(self, tmp_path):
        pybamm.citations._reset()
        with pytest.raises(PybtexError):
            pybamm.citations.read_citations(tmp_path / "missing.bib")

    def test_reload_external_database(self, citation_database, tmp_path):
        citations = pybamm.citations
        citations._reset()
        path = tmp_path / "downstream.bib"
        path.write_text("@misc{Downstream2026, title={Downstream example}}")

        citations.read_citations(path)
        citations.read_citations(path)
        citations.read_citations()

        assert "Downstream2026" in citations._all_citations
        assert "Marquis2019" in citations._all_citations

    def test_external_database_replacement_warning(self, citation_database, tmp_path):
        citations = pybamm.citations
        citations._reset()
        path = tmp_path / "downstream.bib"
        path.write_text("@misc{Downstream2026, title={Original}}")
        citations.read_citations(path)
        path.write_text("@misc{Downstream2026, title={Replacement}}")

        with pytest.warns(UserWarning, match="Replacing citation for Downstream2026"):
            citations.read_citations(path)

        assert "Replacement" in citations._all_citations["Downstream2026"]

    def test_register_loaded_external_citation(
        self, citation_database, tmp_path, capsys
    ):
        citations = pybamm.citations
        citations._reset()
        path = tmp_path / "downstream.bib"
        path.write_text("@misc{Downstream2026, title={Downstream example}}")
        citations.read_citations(path)

        citations.register("Downstream2026")
        citations.print(output_format="bibtex")

        assert "Downstream2026" in citations._papers_to_cite
        assert "Downstream2026" in capsys.readouterr().out

    def test_reasons(self):
        citations = pybamm.citations
        citations._reset()
        citations.register("Marquis2019", reason="SPM model")
        citations.register("Marquis2019", reason="SPM model")
        citations.register("Marquis2019", reason="Parameter values")

        def register_without_class_context():
            citations.register("Marquis2019")
            citations.register("Doyle1993")

        register_without_class_context()

        assert citations._citation_tags["Marquis2019"] == {
            "SPM model",
            "Parameter values",
        }
        assert "Doyle1993" not in citations._citation_tags

    def test_deferred_reasons(self, citation_database):
        citations = pybamm.citations
        citations._reset()
        raw = "@misc{One2026, title={One}}\n@misc{Two2026, title={Two}}"
        citations.register(raw, reason="Downstream model")
        citations.register(raw, reason="Other use")

        citations.print(output_format="bibtex")

        for key in ["One2026", "Two2026"]:
            assert key in citations._papers_to_cite
            assert citations._citation_tags[key] == {
                "Downstream model",
                "Other use",
            }
        assert not citations._unknown_citations

    def test_registration_before_loading(self, citation_database, tmp_path):
        citations = pybamm.citations
        citations._reset()
        citations.register("Downstream2026", reason="Before loading")
        path = tmp_path / "downstream.bib"
        path.write_text("@misc{Downstream2026, title={Example}}")

        citations.read_citations(path)

        assert "Downstream2026" in citations._papers_to_cite
        assert "Downstream2026" not in citations._unknown_citations
        assert citations._citation_tags["Downstream2026"] == {"Before loading"}

    @pytest.mark.parametrize("key_first", [True, False])
    def test_key_and_raw_bibtex_registration_order(self, citation_database, key_first):
        citations = pybamm.citations
        citations._reset()
        key = "Downstream2026"
        raw = "@misc{Downstream2026, title={Example}}"
        registrations = (
            ((key, "Key registration"), (raw, "Raw registration"))
            if key_first
            else ((raw, "Raw registration"), (key, "Key registration"))
        )
        for citation, reason in registrations:
            citations.register(citation, reason=reason)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            citations.print(output_format="bibtex")

        assert not caught
        assert key in citations._papers_to_cite
        assert key not in citations._unknown_citations
        assert citations._citation_tags[key] == {
            "Key registration",
            "Raw registration",
        }
        assert raw not in citations._citation_tags

    @pytest.mark.parametrize("to_file", [False, True])
    def test_verbose_text_output(self, tmp_path, capsys, to_file):
        citations = pybamm.citations
        citations._reset()
        citations.register("Marquis2019", reason="SPM model")
        citations.register("Marquis2019", reason="Parameter values")
        citations.register("Doyle1993", reason="Uncited model")
        citations._papers_to_cite.remove("Doyle1993")
        filename = tmp_path / "citations.txt" if to_file else None

        pybamm.print_citations(filename, verbose=True)

        output = filename.read_text() if to_file else capsys.readouterr().out
        assert "Marquis2019: Parameter values; SPM model" in output
        assert "Doyle1993: Uncited model" not in output
        pybamm.print_citations()
        assert "Citations registered" not in capsys.readouterr().out

    def test_verbose_bibtex_preserves_existing_note(self, citation_database, tmp_path):
        citations = pybamm.citations
        citations._reset()
        raw = "@misc{Example2026, title={Example}, note={Original {note}}}"
        citations.register(raw, reason="SPM model")
        filename = tmp_path / "citations.bib"

        pybamm.print_citations(filename, "bibtex")
        original = filename.read_text()
        pybamm.print_citations(filename, "bibtex", verbose=True)

        assert filename.read_text() == original
        assert "Original {note}" in original

    def test_without_pybtex(self, monkeypatch, tmp_path, caplog):
        citations = pybamm.citations
        citations._reset()
        monkeypatch.setattr(citations, "_module_import_error", True)

        citations.read_citations(tmp_path / "not-read.bib")
        citations.register("Marquis2019", reason="Model")
        citations.register("@misc{Example2026, title={Example}}", reason="Manual")
        citations.print(tmp_path / "not-written.bib", "bibtex", verbose=True)

        assert "Could not print citations" in caplog.text
        assert not (tmp_path / "not-written.bib").exists()
