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
    # Set of citation keys that have been registered
    _papers_to_cite: set
    # Set of unknown citations to parse with pybtex
    _unknown_citations: set
    # Dict mapping citation tags for use when registering citations
    _citation_tags: dict

    def __init__(self):
        self._check_for_bibtex()
        # Dict mapping citations keys to BibTex entries
        self._all_citations: dict[str, str] = dict()

        self.read_citations()
        self._reset()

    def _check_for_bibtex(self):
        try:
            import_optional_dependency("pybtex")
        except ModuleNotFoundError:
            self._module_import_error = True

    def _reset(self):
        """Reset citations to default only (only for testing purposes)"""
        self._papers_to_cite = set()
        self._unknown_citations = set()
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
        if not self._module_import_error:
            parse_file = import_optional_dependency("pybtex.database", "parse_file")
            citations_file = os.path.join(pybamm.__path__[0], "CITATIONS.bib")
            bib_data = parse_file(citations_file, bib_format="bibtex")
            for key, entry in bib_data.entries.items():
                self._add_citation(key, entry)

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
        if not self._module_import_error:
            PybtexError = import_optional_dependency("pybtex.scanner", "PybtexError")
            parse_string = import_optional_dependency("pybtex.database", "parse_string")
            try:
                # Parse string as a bibtex citation, and check that a citation was found
                bib_data = parse_string(key, bib_format="bibtex")
                if not bib_data.entries:
                    raise PybtexError("no entries found")

                # Add and register all citations
                for key, entry in bib_data.entries.items():
                    self._add_citation(key, entry)
                    self._papers_to_cite.add(key)
                return
            except PybtexError as error:
                raise KeyError(
                    f"Not a bibtex citation or known citation: {key}"
                ) from error

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
        be seen where the citations were registered due to the use of PyBaMM models
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
        if not self._module_import_error:
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

            cite_list = self.format_citations(output_format, pybtex)
            self.write_citations(cite_list, filename, verbose)
        else:
            self.print_import_warning()

    def write_citations(self, cite_list, filename, verbose):
        if filename is None:
            print(cite_list)
            if verbose:
                self._tag_citations()  # pragma: no cover
        else:
            with open(filename, "w") as f:
                f.write(cite_list)

    def format_citations(self, output_format, pybtex):
        if output_format == "text":
            cite_list = pybtex.format_from_strings(
                self._cited, style="plain", output_backend="plaintext"
            )
        elif output_format == "bibtex":
            cite_list = "\n".join(self._cited)
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


def print_citations(filename=None, output_format="text", verbose=False):
    """See :meth:`Citations.print`"""
    if verbose and filename is not None:  # pragma: no cover
        raise Exception(
            "Verbose output is available only for the terminal and not for printing to files",
        )
    pybamm.citations.print(filename, output_format, verbose)


citations = Citations()
