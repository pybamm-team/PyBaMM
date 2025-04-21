import pybamm
import os
import warnings
from sys import _getframe
from pybamm.util import import_optional_dependency


class Citations:
    """Entry point to citations management.
    This object may be used to record BibTeX citation information and then register that
    a particular citation is relevant for a particular simulation.

    Citations listed in `pybamm/CITATIONS.bib` can be registered with their citation
    key. For all other works provide a BibTeX Citation to :meth:`register`.

    Examples
    --------
    >>> pybamm.citations.register("Sulzer2021")
    >>> pybamm.citations.register("@misc{Newton1687, title={Mathematical...}}")
    >>> pybamm.print_citations("citations.txt")
    """

    _module_import_error = False

    _all_citations: dict = {}

    def __init__(self):
        self._check_for_bibtex()
        self.read_citations()
        self._reset()

    def _check_for_bibtex(self):
        try:
            import_optional_dependency("bibtexparser")
        except ModuleNotFoundError:
            self._module_import_error = True

    def _reset(self):
        """Reset citations to default only (only for testing purposes)"""
        self._papers_to_cite = set()
        self._unknown_citations = set()
        self._citation_tags = {}
        # Register the PyBaMM paper and the NumPy paper
        self.register("Sulzer2021")
        self.register("Harris2020")

    @staticmethod
    def _caller_name():
        """
        Returns the qualified name of classes that call :meth:`register` internally.
        Gets cached in order to reduce the number of calls.
        """
        caller_name = _getframe().f_back.f_back.f_locals["self"].__class__.__qualname__
        return caller_name

    def read_citations(self):
        """Reads the citations in `pybamm.CITATIONS.bib`. Other works can be cited
        by passing a BibTeX citation to :meth:`register`.
        """
        if not self._module_import_error:
            bibtexparser = import_optional_dependency("bibtexparser")
            citations_file = os.path.join(pybamm.__path__[0], "CITATIONS.bib")
            library = bibtexparser.parse_file(citations_file)
            for entry in library.entries:
                self._add_citation(entry.key, entry.fields_dict)

    def _add_citation(self, key, entry):
        """Adds `entry` to `self._all_citations` under `key`, warning the user if a
        previous entry is overwritten
        """
        if not isinstance(key, str):
            raise TypeError(f"Expected citation key as str, got {type(key).__name__}")
        if not isinstance(entry, dict):
            try:
                entry = dict(entry)
            except Exception as e:
                raise TypeError(
                    f"Expected citation entry as dict, got {type(entry).__name__}"
                ) from e

        # Warn if overwriting a previous citation
        if key in self._all_citations:
            existing = self._string_formatting(self._all_citations[key])
            new = self._string_formatting(entry)
            if existing != new:
                warnings.warn(f"Replacing citation for {key}", stacklevel=2)

        # Add to database
        self._all_citations[key] = entry

    def _string_formatting(self, entry):
        fields = entry
        authors = self._format_author(str(fields.get("author", "")))
        title = self._format_title(str(fields.get("title", "")))
        journal = self._format_journal(str(fields.get("journal", "")))
        volume = str(fields.get("volume", ""))
        number = str(fields.get("number", ""))
        pages = str(fields.get("pages", ""))
        year = str(fields.get("year", ""))
        doi = str(fields.get("doi", ""))

        parts = [f"{authors}"]
        if title:
            parts.append(f'"{title}"')
        if journal:
            vol_issue = f"{volume}({number})" if volume and number else volume or number
            journal_str = f"*{journal}*        {vol_issue}".strip()
            parts.append(journal_str)
        if year:
            parts.append(year)
        if pages:
            parts[-1] += f", {pages}"
        if doi:
            parts.append(f"doi:{doi}")

        return ". ".join(filter(None, parts)) + "."

    def _format_author(self, author_str):
        authors = author_str.replace("\n", " ").split(" and ")
        formatted = []
        for author in authors:
            parts = author.split(", ")
            if len(parts) > 1:
                formatted.append(f"{parts[1]} {parts[0]}")
            else:
                formatted.append(parts[0])
        return ", ".join(formatted) + "." if authors else ""

    def _format_title(self, title):
        return str(title).replace("\n", " ").strip()

    def _format_journal(self, journal):
        return str(journal).replace("\n", " ").strip()

    def register(self, key):
        """Register a paper to be cited.
            Accepts either a citation key from pybamm/CITATIONS.bib or a raw BibTeX entry.
        When registering unknown citations, validates BibTeX syntax using bibtexparser.

        Parameters
        ----------
        key : str
            - For pre-defined citations: Exact key from CITATIONS.bib (e.g. "Sulzer2021")
            - For custom citations: Complete BibTeX entry (e.g. "@article{...}")

        Raises
        ------
        KeyError
            If provided BibTeX is malformed or key doesn't exist in CITATIONS.bib
        TypeError
            If input is not a string

        Warns
        -----
        UserWarning
            When overwriting an existing citation with different content
        """

        if key in self._all_citations:
            self._papers_to_cite.add(key)
            try:
                caller = self._caller_name()
                self._citation_tags[key] = caller
            except KeyError:
                pass
        else:
            try:
                bibtexparser = import_optional_dependency("bibtexparser")
                library = bibtexparser.parse_string(key)
                if len(library.failed_blocks) > 0 or not library.entries:
                    raise ValueError("No entries found in BibTeX string")
                entry = library.entries[0]
                citation_id = entry.key
                if not citation_id:
                    raise ValueError("Missing ID in BibTeX entry")
                self._add_citation(citation_id, entry.fields_dict)
                self._papers_to_cite.add(citation_id)
            except Exception as e:
                self._unknown_citations.add(key)
                raise KeyError(f"Invalid BibTeX entry: {e!s}") from e

    def print(self, filename=None, output_format="text", verbose=False):
        """
            Print all registered citations in the desired format.

        This method outputs all citations that have been registered during the current
        session, either to the terminal or to a specified file. Citations can be formatted
        as plain text (suitable for inclusion in manuscripts) or as BibTeX entries
        (for use in reference managers).

        Parameters
        ----------
        filename : str or Path, optional
            The file path to which citations will be written. If None (default),
            citations are printed to the terminal (stdout).
        output_format : str, optional
            The output format for citations. Must be either "text" (default) for
            human-readable references, or "bibtex" for raw BibTeX entries.
        verbose : bool, optional
            If True, prints additional information about where each citation was
            registered (e.g., which model or solver triggered the citation).
            Verbose output is only available when printing to the terminal.

        Raises
        ------
        pybamm.OptionError
            If an invalid output_format is provided (not "text" or "bibtex").
        Exception
            If verbose output is requested when writing to a file.
        """
        if self._module_import_error:
            self.print_import_warning()
            return

        citations = []
        for key in self._papers_to_cite:
            if entry := self._all_citations.get(key):
                if output_format == "text":
                    citations.append(self._string_formatting(entry))
                elif output_format == "bibtex":
                    bibtexparser = import_optional_dependency("bibtexparser")
                    from bibtexparser.library import Library
                    from bibtexparser.model import Entry

                    dummy_lib = Library()
                    fields = entry
                    bib_entry = Entry("article", key, fields=fields)
                    dummy_lib.add_entry(bib_entry)
                    citations.append(bibtexparser.write_string(dummy_lib))

        output = (
            "\n\n".join(citations) if output_format == "text" else "\n".join(citations)
        )

        if filename:
            with open(filename, "w") as f:
                f.write(output)
            if verbose:
                self._tag_citations(filename)
        else:
            print(output)
            if verbose:
                self._tag_citations()

    def _tag_citations(self, filename=None):
        if self._citation_tags:
            msg = "\nCitations registered:\n" + "\n".join(
                f"{k} was cited due to {v}" for k, v in self._citation_tags.items()
            )
            if filename:
                with open(filename, "a") as f:
                    f.write(msg)
            else:
                print(msg)

    def print_import_warning(self):
        if self._module_import_error:
            pybamm.logger.warning(
                "Could not print citations because the 'bibtexparser' library is not installed. "
                "Please, install 'pybamm[cite]' to print citations."
            )


def print_citations(filename=None, output_format="text", verbose=False):
    """See :meth:`Citations.print`"""
    if verbose and filename is not None:  # pragma: no cover
        raise Exception(
            "Verbose output is available only for the terminal and not for printing to files",
        )
    pybamm.citations.print(filename, output_format, verbose)


citations = Citations()
