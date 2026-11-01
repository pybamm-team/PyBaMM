from __future__ import annotations

import os
import warnings
from sys import _getframe
from types import ModuleType

import pybamm
from pybamm.util import import_optional_dependency


class Citations:
    """Entry point to citations management.
    This object may be used to record BibTeX citation information and then register that
    a particular citation is relevant for a particular simulation.

    Citations listed in `pybamm/CITATIONS.bib` can be registered with their citation
    key. Load other databases with :meth:`read_citations`, or provide a BibTeX
    citation to :meth:`register`. Requires the optional ``pybamm[cite]`` extra.

    Examples
    --------
    >>> pybamm.citations.register("Sulzer2021")
    >>> pybamm.citations.register("@misc{Newton1687, title={Mathematical...}}")
    >>> pybamm.print_citations("citations.txt")
    """

    _module_import_error = False
    # Set of citation keys that have been registered
    _papers_to_cite: set
    # Set of unknown citations to parse with pybtex
    _unknown_citations: set
    # Dict mapping citation tags for use when registering citations
    _citation_tags: dict[str, set[str]]

    def __init__(self):
        self._check_for_bibtex()
        # Dict mapping citations keys to BibTex entries
        self._all_citations: dict[str, str] = {}

        self._reset()
        self.read_citations()

    def _check_for_bibtex(self):
        try:
            import_optional_dependency("pybtex")
        except ModuleNotFoundError:
            self._module_import_error = True

    def _reset(self):
        """Reset citations to default only (only for testing purposes)"""
        self._papers_to_cite = set()
        self._unknown_citations = set()
        self._citation_tags = {}
        # Register the PyBaMM paper and the NumPy paper
        self.register("Sulzer2021", reason="PyBaMM")
        self.register("Harris2020", reason="NumPy")

    @staticmethod
    def _caller_name():
        """Return the qualified name of the class calling :meth:`register`."""
        caller_name = _getframe().f_back.f_back.f_locals["self"].__class__.__qualname__
        return caller_name

    def read_citations(self, filename: str | os.PathLike | None = None) -> None:
        """Load known citations from a BibTeX file without registering them.

        Parameters
        ----------
        filename : str or pathlib.Path, optional
            Database to load. Defaults to PyBaMM's bundled ``CITATIONS.bib``.
        """
        if not self._module_import_error:
            parse_file = import_optional_dependency("pybtex.database", "parse_file")
            if filename is None:
                filename = os.path.join(pybamm.__path__[0], "CITATIONS.bib")
            bib_data = parse_file(filename, bib_format="bibtex")
            for key, entry in bib_data.entries.items():
                self._add_citation(key, entry)
                if key in self._unknown_citations:
                    self._papers_to_cite.add(key)
                    self._unknown_citations.discard(key)

    def _add_citation(self, key, entry):
        """Adds `entry` to `self._all_citations` under `key`, warning the user if a
        previous entry is overwritten
        """
        if not self._module_import_error:
            Entry = import_optional_dependency("pybtex.database", "Entry")
            # Check input types are correct
            if not isinstance(key, str) or not isinstance(entry, Entry):
                raise TypeError()

            # Warn if overwriting a previous citation
            new_citation = entry.to_string("bibtex")
            if key in self._all_citations and new_citation != self._all_citations[key]:
                warnings.warn(f"Replacing citation for {key}", stacklevel=2)

            # Add to database
            self._all_citations[key] = new_citation

    def _add_citation_tag(self, key, entry):
        """Add a distinct reason for citing a key or deferred BibTeX string."""
        self._citation_tags.setdefault(key, set()).add(entry)

    @property
    def _cited(self):
        """Return a list of the BibTeX entries that have been cited"""
        return [self._all_citations[key] for key in self._papers_to_cite]

    def register(self, key: str, reason: str | None = None) -> None:
        """Register a paper to be cited, one at a time. The intended use is that
        :meth:`register` should be called only when the referenced functionality is
        actually being used.

        .. warning::
            Registering a BibTeX citation, with the same key as an existing citation,
            will overwrite the current citation.

        Parameters
        ----------
        key : str
            - A known citation key (including loaded external databases), or
            - A BibTeX formatted citation
        reason : str, optional
            Human-readable reason for citing the reference. Defaults to the calling
            class name, if available. Distinct reasons accumulate for each citation.

        Raises
        ------
        TypeError
            If ``reason`` is neither a string nor None.
        """
        if reason is not None and not isinstance(reason, str):
            raise TypeError("reason must be a string or None")
        if reason is None:
            try:
                reason = Citations._caller_name()
            except KeyError:
                # Standalone callers need not provide a class context or reason.
                pass
        if reason is not None:
            self._add_citation_tag(key, reason)

        # Check if citation is a known key
        if key in self._all_citations:
            self._papers_to_cite.add(key)
            self._unknown_citations.discard(key)
        else:
            self._unknown_citations.add(key)
            return

    def _parse_citation(self, key):
        """
        Parses a citation with pybtex and adds it to the _papers_to_cite set. This
        method is called when a citation is unknown at the time of registration.

        Parameters
        ----------
        key: str
            A BibTeX formatted citation
        """
        if not self._module_import_error:
            if key in self._all_citations:
                self._papers_to_cite.add(key)
                self._unknown_citations.discard(key)
                return
            PybtexError = import_optional_dependency("pybtex.scanner", "PybtexError")
            parse_string = import_optional_dependency("pybtex.database", "parse_string")
            try:
                bib_data = parse_string(key, bib_format="bibtex")
                if not bib_data.entries:
                    raise PybtexError("no entries found")

                # Add and register all citations
                for entry_key, entry in bib_data.entries.items():
                    self._add_citation(entry_key, entry)
                    self._papers_to_cite.add(entry_key)
                    for reason in self._citation_tags.get(key, ()):
                        self._add_citation_tag(entry_key, reason)
                self._citation_tags.pop(key, None)
                self._unknown_citations.discard(key)
                return
            except PybtexError as error:
                raise KeyError(
                    f"Not a bibtex citation or known citation: {key}"
                ) from error

    def _tag_citations(self):
        """Format citation reasons in a stable order."""
        lines = [
            f"{key}: {'; '.join(sorted(reasons))}"
            for key, reasons in sorted(self._citation_tags.items())
            if key in self._papers_to_cite
        ]
        return "\nCitations registered: \n\n" + "\n".join(lines) if lines else ""

    def print(
        self,
        filename: str | os.PathLike | None = None,
        output_format: str = "text",
        verbose: bool = False,
    ) -> None:
        """Print citations with optional reasons.

        .. warning::
            Invalid citations produce a warning.

        Parameters
        ----------
        filename : str or pathlib.Path, optional
            Filename to which to print citations. If None, citations are printed
            to the terminal.
        output_format : str, optional
            Bibliography format: ``"text"`` (default) or ``"bibtex"``.
        verbose: bool, optional
            If True, append reasons to text output or BibTeX note fields, for both
            terminal and file output. Existing notes are preserved. Defaults to False.

        Examples
        --------
        .. code-block:: python

            pybamm.lithium_ion.SPM()
            pybamm.print_citations(verbose=True)

        will append the following at the end of the list of citations:

        .. code-block::

            Citations registered:

            Marquis2019: SPM

        """
        # Parse citations that were not known keys at registration, but do not
        # fail if they cannot be parsed
        if not self._module_import_error:
            pybtex = import_optional_dependency("pybtex")
            for key in tuple(self._unknown_citations):
                try:
                    self._parse_citation(key)
                except KeyError:
                    # Another pending BibTeX entry may define this key before we warn.
                    pass
            # Raw BibTeX can resolve a key that failed earlier in this pass.
            for key in tuple(self._unknown_citations):
                if key not in self._all_citations:
                    warnings.warn(
                        message=f'\nCitation with key "{key}" is invalid. Please try again\n',
                        category=UserWarning,
                        stacklevel=2,
                    )
                    self._citation_tags.pop(key, None)
                self._unknown_citations.discard(key)

            cite_list = self.format_citations(output_format, pybtex, verbose)
            self.write_citations(
                cite_list, filename, verbose and output_format == "text"
            )
        else:
            self.print_import_warning()

    def write_citations(
        self, cite_list: str, filename: str | os.PathLike | None, verbose: bool
    ) -> None:
        if verbose:
            cite_list += self._tag_citations()
        if filename is None:
            print(cite_list)
        else:
            with open(filename, "w") as f:
                f.write(cite_list)

    def format_citations(
        self, output_format: str, pybtex: ModuleType, verbose: bool = False
    ) -> str:
        if output_format == "text":
            cite_list = pybtex.format_from_strings(
                self._cited, style="plain", output_backend="plaintext"
            )
        elif output_format == "bibtex":
            cite_list = "\n".join(self._cited)
            if verbose:
                parse_string = import_optional_dependency(
                    "pybtex.database", "parse_string"
                )
                Writer = import_optional_dependency(
                    "pybtex.database.output.bibtex", "Writer"
                )

                class CitationWriter(Writer):
                    def _encode(self, text):
                        # Parsed fields already contain LaTeX escapes.
                        return text

                bibliography = parse_string(cite_list, "bibtex")
                for key, entry in bibliography.entries.items():
                    reasons = self._citation_tags.get(key)
                    if reasons:
                        annotation = "Cited for: " + "; ".join(sorted(reasons))
                        annotation = annotation.translate(
                            str.maketrans(
                                {
                                    "\\": r"\textbackslash{}",
                                    "{": r"\textbraceleft{}",
                                    "}": r"\textbraceright{}",
                                    "&": r"\&",
                                    "%": r"\%",
                                    "_": r"\_",
                                    "#": r"\#",
                                    "$": r"\$",
                                    "^": r"\textasciicircum{}",
                                    "~": r"\textasciitilde{}",
                                }
                            )
                        )
                        note = entry.fields.get("note")
                        entry.fields["note"] = (
                            f"{note}. {annotation}" if note else annotation
                        )
                cite_list = CitationWriter().to_string(bibliography)
        else:
            raise pybamm.OptionError(
                f"Output format {output_format} not recognised."
                "It should be 'text' or 'bibtex'."
            )
        return cite_list

    def print_import_warning(self):
        if self._module_import_error:
            pybamm.logger.warning(
                "Could not print citations because the 'pybtex' library is not installed. "
                "Please, install 'pybamm[cite]' to print citations."
            )


def print_citations(
    filename: str | os.PathLike | None = None,
    output_format: str = "text",
    verbose: bool = False,
) -> None:
    """See :meth:`Citations.print`"""
    pybamm.citations.print(filename, output_format, verbose)


citations = Citations()
