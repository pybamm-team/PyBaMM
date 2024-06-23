#
# Bibliographical information for PyBaMM
# Inspired by firedrake/PETSc citation workflow
# https://firedrakeproject.org/citing.html
#
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

    def __init__(self):
        # Set of citation keys that have been registered
        self._papers_to_cite = set()

        # Dict mapping citations keys to BibTex entries
        self._all_citations: dict[str, str] = dict()

        # Set of unknown citations to parse with pybtex
        self._unknown_citations = set()

        # Dict mapping citation tags for use when registering citations
        self._citation_tags = dict()

        self.read_citations()
        self._reset()

    def _reset(self):
        """Reset citations to default only (only for testing purposes)"""
        # Initialize empty papers to cite
        self._papers_to_cite = set()
        # Initialize empty set of unknown citations
        self._unknown_citations = set()
        # Initialize empty citation tags
        self._citation_tags = dict()
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
        try:
            parse_file = import_optional_dependency("pybtex.database", "parse_file")
            citations_file = os.path.join(pybamm.root_dir(), "pybamm", "CITATIONS.bib")
            bib_data = parse_file(citations_file, bib_format="bibtex")
            for key, entry in bib_data.entries.items():
                self._add_citation(key, entry)
        except ModuleNotFoundError:  # pragma: no cover
            pybamm.logger.warning(
                "Citations could not be read because the 'pybtex' library is not installed. "
                "Install 'pybamm[cite]' to enable citation reading."
            )

    def _add_citation(self, key, entry):
        """Adds `entry` to `self._all_citations` under `key`, warning the user if a
        previous entry is overwritten
        """

        try:
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
        except ModuleNotFoundError:  # pragma: no cover
            pybamm.logger.warning(
                f"Could not add citation for '{key}' because the 'pybtex' library is not installed. "
                "Install 'pybamm[cite]' to enable adding citations."
            )

    def _add_citation_tag(self, key, entry):
        """Adds a tag for a citation key in the dict, which represents the name of the
        class that called :meth:`register`"""
        self._citation_tags[key] = entry

    @property
    def _cited(self):
        """Return a list of the BibTeX entries that have been cited"""
        return [self._all_citations[key] for key in self._papers_to_cite]

    def register(self, key):
        """Register a paper to be cited, one at a time. The intended use is that
        :meth:`register` should be called only when the referenced functionality is
        actually being used.

        .. warning::
            Registering a BibTeX citation, with the same key as an existing citation,
            will overwrite the current citation.

        Parameters
        ----------
        key : str
            - The citation key for an entry in `pybamm/CITATIONS.bib` or
            - A BibTeX formatted citation
        """
        # Check if citation is a known key
        if key in self._all_citations:
            self._papers_to_cite.add(key)
            # Add citation tags for the key for verbose output, but
            # don't if they already exist in _citation_tags dict
            if key not in self._citation_tags:
                try:
                    caller = Citations._caller_name()
                    self._add_citation_tag(key, entry=caller)
                    # Don't add citation tags if the citation is registered manually
                except KeyError:  # pragma: no cover
                    pass
        else:
            # If citation is unknown, parse it later with pybtex
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
        try:
            PybtexError = import_optional_dependency("pybtex.scanner", "PybtexError")
            parse_string = import_optional_dependency("pybtex.database", "parse_string")
            try:
                # Parse string as a bibtex citation, and check that a citation was found
                bib_data = parse_string(key, bib_format="bibtex")
                if not bib_data.entries:
                    raise PybtexError("no entries found")

                # Add and register all citations
                for key, entry in bib_data.entries.items():
                    # Add to _all_citations dictionary
                    self._add_citation(key, entry)
                    # Add to _papers_to_cite set
                    self._papers_to_cite.add(key)
                    return
            except PybtexError as error:
                # Unable to parse / unknown key
                raise KeyError(
                    f"Not a bibtex citation or known citation: {key}"
                ) from error
        except ModuleNotFoundError:  # pragma: no cover
            pybamm.logger.warning(
                f"Could not parse citation for '{key}' because the 'pybtex' library is not installed. "
                "Install 'pybamm[cite]' to enable citation parsing."
            )

    def _tag_citations(self):
        """Prints the citation tags for the citations that have been registered
        (non-manually) in the code, for verbose output purposes
        """
        if self._citation_tags:  # pragma: no cover
            print("\nCitations registered: \n")
            for key, entry in self._citation_tags.items():
                print(f"{key} was cited due to the use of {entry}")

    def print(self, filename=None, output_format="text", verbose=False):
        """Print all citations that were used for running simulations. The verbose
        option is provided to print tags for citations in the output such that it can
        can be seen where the citations were registered due to the use of PyBaMM models
        and solvers in the code.

        .. note::
            If a citation is registered manually, it will not be tagged.

        .. warning::
            This function will notify the user if a citation that has been previously
            registered is invalid or cannot be parsed.

        Parameters
        ----------
        filename : str, optional
            Filename to which to print citations. If None, citations are printed
            to the terminal.
        verbose: bool, optional
            If True, prints the citation tags for the citations that have been
            registered. An example of the output is shown below.

        Examples
        --------
        .. code-block:: python

            pybamm.lithium_ion.SPM()
            pybamm.Citations.print(verbose=True) or pybamm.print_citations(verbose=True)

        will append the following at the end of the list of citations:

        .. code-block::

            Citations registered:

            Marquis2019 was cited due to the use of SPM

        """
        # Parse citations that were not known keys at registration, but do not
        # fail if they cannot be parsed
        try:
            pybtex = import_optional_dependency("pybtex")
            try:
                for key in self._unknown_citations:
                    self._parse_citation(key)
            except KeyError:  # pragma: no cover
                warnings.warn(
                    message=f'\nCitation with key "{key}" is invalid. Please try again\n',
                    category=UserWarning,
                    stacklevel=2,
                )
                # delete the invalid citation from the set
                self._unknown_citations.remove(key)

            if output_format == "text":
                citations = pybtex.format_from_strings(
                    self._cited, style="plain", output_backend="plaintext"
                )
            elif output_format == "bibtex":
                citations = "\n".join(self._cited)
            else:
                raise pybamm.OptionError(
                    f"Output format {output_format} not recognised."
                    "It should be 'text' or 'bibtex'."
                )

            if filename is None:
                print(citations)
                if verbose:
                    self._tag_citations()  # pragma: no cover
            else:
                with open(filename, "w") as f:
                    f.write(citations)
        except ModuleNotFoundError:  # pragma: no cover
            pybamm.logger.warning(
                "Could not print citations because the 'pybtex' library is not installed. "
                "Please, install 'pybamm[cite]' to print citations."
            )


def print_citations(filename=None, output_format="text", verbose=False):
    """See :meth:`Citations.print`"""
    if verbose:  # pragma: no cover
        if filename is not None:  # pragma: no cover
            raise Exception(
                "Verbose output is available only for the terminal and not for printing to files",
            )
        else:
            citations.print(filename, output_format, verbose=True)
    else:
        pybamm.citations.print(filename, output_format)


citations = Citations()
