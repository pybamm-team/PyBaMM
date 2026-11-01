import contextlib
import io
import sys
import warnings

import pytest
from pybtex.database import Entry, parse_string
from pybtex.exceptions import PybtexError

import pybamm


@pytest.fixture(autouse=True)
def citations():
    pybamm.citations._reset()
    yield pybamm.citations
    pybamm.citations._reset()


@pytest.fixture
def citation_database(citations, monkeypatch):
    # Resetting registrations does not restore added or replaced bibliography entries.
    monkeypatch.setattr(citations, "_all_citations", citations._all_citations.copy())


class TestCitations:
    def test_default_reasons(self):
        citations = pybamm.Citations()
        for _ in range(2):
            assert citations._citation_tags == {
                "Sulzer2021": {"PyBaMM"},
                "Harris2020": {"NumPy"},
            }
            citations._reset()

    @pytest.mark.parametrize("as_string", [False, True])
    def test_external_database(
        self, citations, citation_database, tmp_path, capsys, as_string
    ):
        path = tmp_path / "downstream.bib"
        path.write_text("@misc{Downstream2026, title={Downstream example}}")
        registered = citations._papers_to_cite.copy()
        citations.read_citations(str(path) if as_string else path)
        assert "Downstream2026" in citations._all_citations
        assert citations._papers_to_cite == registered
        assert "Marquis2019" in citations._all_citations
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            citations.read_citations(path)
            citations.read_citations()
        assert not caught
        citations.register("Downstream2026")
        citations.print(output_format="bibtex")
        assert "Downstream2026" in capsys.readouterr().out
        path.write_text("@misc{Downstream2026, title={Replacement}}")
        with pytest.warns(UserWarning, match="Replacing citation for Downstream2026"):
            citations.read_citations(path)
        with pytest.raises(PybtexError):
            citations.read_citations(tmp_path / "missing.bib")

    def test_reasons(self, citations, monkeypatch):
        def unexpected_caller():
            pytest.fail("Explicit reasons must not inspect the caller")

        with monkeypatch.context() as patch:
            patch.setattr(pybamm.Citations, "_caller_name", unexpected_caller)
            citations.register("Marquis2019", reason="SPM model")
            citations.register("Marquis2019", reason="SPM model")
            citations.register("Marquis2019", reason="Parameter values")
        citations.register("Marquis2019")
        assert citations._citation_tags["Marquis2019"] == {
            "SPM model",
            "Parameter values",
            type(self).__qualname__,
        }
        namespace = {"register": citations.register}
        exec("register('Marquis2019'); register('Doyle1993')", namespace)
        assert len(citations._citation_tags["Marquis2019"]) == 3
        assert "Doyle1993" not in citations._citation_tags
        with pytest.raises(TypeError, match="reason"):
            citations.register("Marquis2019", reason=123)
        citations._reset()
        assert "Marquis2019" not in citations._citation_tags
        assert not citations._unknown_citations

    def test_deferred_reasons(self, citations, citation_database, capsys):
        raw = "@misc{One2026, title={One}}\n@misc{Two2026, title={Two}}"
        citations.register(raw, reason="Downstream model")
        citations.register(raw, reason="Other use")
        citations.print(output_format="bibtex", verbose=True)
        entries = parse_string(capsys.readouterr().out, "bibtex").entries
        for key in ["One2026", "Two2026"]:
            assert entries[key].fields["note"] == (
                "Cited for: Downstream model; Other use"
            )
        assert not citations._unknown_citations
        citations.register(raw, reason="Later use")
        citations.print(output_format="bibtex", verbose=True)
        entries = parse_string(capsys.readouterr().out, "bibtex").entries
        assert "Later use" in entries["One2026"].fields["note"]
        citations.register(raw, reason="Pending use")
        citations._reset()
        assert raw not in citations._citation_tags
        assert not citations._unknown_citations

    def test_unmatched_braces_in_reason(self, citations, capsys):
        citations.register("Marquis2019", reason=r"Literal { and \ path")
        citations.print(output_format="bibtex", verbose=True)
        entry = parse_string(capsys.readouterr().out, "bibtex").entries["Marquis2019"]
        assert entry.fields["note"] == (
            r"Cited for: Literal \textbraceleft{} and \textbackslash{} path"
        )

    def test_registration_before_loading(
        self, citations, citation_database, tmp_path, capsys
    ):
        citations.register("Downstream2026", reason="Before loading")
        path = tmp_path / "downstream.bib"
        path.write_text("@misc{Downstream2026, title={Example}}")
        citations.read_citations(path)
        assert "Downstream2026" in citations._papers_to_cite
        assert "Downstream2026" not in citations._unknown_citations
        citations.print(output_format="bibtex", verbose=True)
        entries = parse_string(capsys.readouterr().out, "bibtex").entries
        assert entries["Downstream2026"].fields["note"] == "Cited for: Before loading"

    @pytest.mark.parametrize("key_first", [True, False])
    def test_pending_key_and_bibtex_reasons(
        self, citations, citation_database, capsys, key_first
    ):
        key = "Downstream2026"
        raw = "@misc{Downstream2026, title={Example}}"
        citations.register(key, reason="Key registration")
        citations.register(raw, reason="Raw registration")
        order = (key, raw) if key_first else (raw, key)

        class PendingCitations(set):
            def __iter__(self):
                return (citation for citation in order if citation in self)

        # Exercise both possible traversal orders of the pending set.
        citations._unknown_citations = PendingCitations(citations._unknown_citations)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            citations.print(output_format="bibtex", verbose=True)
        assert not caught
        entries = parse_string(capsys.readouterr().out, "bibtex").entries
        assert entries[key].fields["note"] == (
            "Cited for: Key registration; Raw registration"
        )
        assert not citations._unknown_citations
        assert raw not in citations._citation_tags

    @pytest.mark.parametrize("output_format", ["text", "bibtex"])
    @pytest.mark.parametrize("to_file", [False, True])
    def test_annotated_output(
        self, citations, citation_database, tmp_path, capsys, output_format, to_file
    ):
        raw = r"@misc{Example2026, title={Example}, note={Original {note} \& details}}"
        reason = 'Model {A} & parameters: 50%_# "quoted"'
        citations.register(raw, reason=reason)
        citations.register(raw, reason="Another reason")
        filename = tmp_path / "citations" if to_file else None

        def export(verbose):
            pybamm.print_citations(filename, output_format, verbose=verbose)
            return filename.read_text() if to_file else capsys.readouterr().out

        first = export(True)
        source = citations._all_citations.copy()
        original_note = (
            parse_string(source["Example2026"], "bibtex")
            .entries["Example2026"]
            .fields["note"]
        )
        assert export(True) == first
        assert first.count("Another reason") == 1
        if output_format == "bibtex":
            entries = parse_string(first, "bibtex").entries
            assert entries["Example2026"].fields["note"] == (
                original_note
                + r". Cited for: Another reason; Model \textbraceleft{}A\textbraceright{} "
                r'\& parameters: 50\%\_\# "quoted"'
            )
        else:
            assert "Example2026: Another reason; " + reason in first
        citations.register("Example2026", reason="Later reason")
        assert "Later reason" in export(True)
        plain = export(False)
        assert "Another reason" not in plain
        assert citations._all_citations == source
        if output_format == "bibtex":
            assert parse_string(plain, "bibtex").entries["Example2026"].fields[
                "note"
            ] == (original_note)
        with pytest.raises(pybamm.OptionError, match="'text' or 'bibtex'"):
            pybamm.print_citations(filename, "invalid", verbose=True)

    def test_without_pybtex(self, monkeypatch, tmp_path, caplog):
        for name in list(sys.modules):
            if name == "pybtex" or name.startswith("pybtex."):
                monkeypatch.setitem(sys.modules, name, None)
        # Construction must exercise imports while pybtex is unavailable.
        citations = pybamm.Citations()
        citations.read_citations(tmp_path / "not-read.bib")
        citations.register("Marquis2019", reason="Model")
        citations.register("@misc{Example2026, title={Example}}", reason="Manual")
        citations.print(tmp_path / "not-written.bib", "bibtex", verbose=True)
        assert "Could not print citations" in caplog.text
        assert not (tmp_path / "not-written.bib").exists()

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

    def test_overwrite_citation(self, citation_database):
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
