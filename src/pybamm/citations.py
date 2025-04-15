import pybamm
import os
import warnings
from sys import _getframe
from bibtexparser import BibDatabase, UndefinedString
from bibtexparser import BibTexParser
from bibtexparser import BibTexWriter
from bibtexparser import load as bibtex_load
from bibtexparser.model import Entry
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
            citations_file = os.path.join(pybamm.__path__[0], "CITATIONS.bib")
            with open(citations_file) as bibtex_file:
              parser = BibTexParser(common_strings=True)
              bib_data = bibtex_load(bibtex_file, parser=parser)
              for entry in bib_data.entries:
                  self._add_citation(entry.key, entry)

    def _add_citation(self, key, entry):
        """Adds `entry` to `self._all_citations` under `key`, warning the user if a
        previous entry is overwritten
        """
        if not isinstance(key, str) or not isinstance(entry, Entry):
                raise TypeError()

            # Warn if overwriting a previous citation
        if key in self._all_citations:
                existing = self._string_formating(self._all_citations[key])
                new = self._string_formating(entry)
                if existing != new:
                 warnings.warn(f"Replacing citation for {key}", stacklevel=2)

            # Add to database
        self._all_citations[key] = entry

    def _string_formatting(self, entry):
        """Custom formatting for bibtexparser entries"""
        fields = entry.fields_dict
        authors = self._format_author(fields.get("author", ""))
        title = self._format_title(fields.get("title", ""))
        journal = self._format_journal(fields.get("journal", ""))
        volume = fields.get("volume", "")
        number = f"({fields.get('number', '')})" if fields.get("number") else ""
        pages = fields.get("pages", "")
        year = fields.get("year", "")
        doi = f"doi:{fields.get('doi', '')}" if fields.get("doi") else ""
        
        components = [
            f"{authors} {title}",
            f"{journal} {volume}{number}",
            f"{year}{', ' + pages if pages else ''}",
            doi
        ]
        return ". ".join(filter(None, components)).replace("  ", " ") + "."  
    
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
        title = title.replace("\n", " ").strip()
        return f'"{title}"' if title else ""

    def _format_journal(self, journal):
        journal = journal.replace("\n", " ").strip()
        return f"*{journal}*" if journal else ""

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

        # Check if citation is a known key
        if key in self._all_citations:
            self._papers_to_cite.add(key)

        else:
            try:
                parser = BibTexParser(common_strings=True)
                bib_db = bibtex_load(key, parser=parser)
                if not bib_db.entries:
                    raise ValueError("No entries found in BibTeX string")
                entry = bib_db.entries[0]
                self._add_citation(entry.key, entry)
                self._papers_to_cite.add(entry.key)
            except (UndefinedString, IndexError, TypeError) as e:
            # If citation is unknown, parse it later with pybtex
             self._unknown_citations.add(key)
             raise KeyError(f"Invalid BibTeX entry: {str(e)}") from e

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
                    writer = BibTexWriter()
                    citations.append(writer.write(BibDatabase(entries=[entry])))
        
        output = "\n\n".join(citations) if output_format == "text" else "\n".join(citations)
        
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
                f"{k} was cited due to {v}" for k,v in self._citation_tags.items()
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
