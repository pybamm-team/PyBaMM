#
# Bibliographical information for PyBaMM
# Inspired by firedrake/PETSc citation workflow
# https://www.firedrakeproject.org/citing.html
#
import pybamm
import os
import warnings
import pybtex
from inspect import stack
from pybtex.database import parse_file, parse_string, Entry
from pybtex.scanner import PybtexError


class Citations:

    """Entry point to citations management.
    This object may be used to record BibTeX citation information and then register that
    a particular citation is relevant for a particular simulation.

    Citations listed in `pybamm/CITATIONS.txt` can be registered with their citation
    key. For all other works provide a BibTeX Citation to :meth:`register`.

    Examples
    --------
    >>> import pybamm
    >>> pybamm.citations.register("Sulzer2021")
    >>> pybamm.citations.register("@misc{Newton1687, title={Mathematical...}}")
    >>> pybamm.print_citations("citations.txt")
    """

    def __init__(self):
        # Set of citation keys that have been registered
        self._papers_to_cite = set()

        # Dict mapping citations keys to BibTex entries
        self._all_citations: dict[str, str] = dict()

        # Dict mapping citation tags for use when registering citations
        self._citation_tags = dict()

        # store citation error
        self._citation_err_msg = None

        try:
            self.read_citations()
            self._reset()
        except Exception as e:  # pragma: no cover
            self._citation_err_msg = e

    def _reset(self):
        """Reset citations to default only (only for testing purposes)"""
        # Initialize empty papers to cite
        self._papers_to_cite = set()
        # Initialize empty citation tags
        self._citation_tags = dict()
        # Register the PyBaMM paper and the numpy paper
        self.register("Sulzer2021")
        self.register("Harris2020")

    def _caller_name():
        """
        Returns the qualified name of classes that call :meth:`register` internally.
        This is used for tagging citations but only for verbose output
        """
        # Attributed to https://stackoverflow.com/a/17065634
        caller_name = stack()[2][0].f_locals["self"].__class__.__qualname__
        return caller_name

    def read_citations(self):
        """Reads the citations in `pybamm.CITATIONS.txt`. Other works can be cited
        by passing a BibTeX citation to :meth:`register`.
        """
        citations_file = os.path.join(pybamm.root_dir(), "pybamm", "CITATIONS.txt")
        bib_data = parse_file(citations_file, bib_format="bibtex")
        for key, entry in bib_data.entries.items():
            self._add_citation(key, entry)

    def _add_citation(self, key, entry):
        """Adds `entry` to `self._all_citations` under `key`, warning the user if a
        previous entry is overwritten
        """

        # Check input types are correct
        if not isinstance(key, str) or not isinstance(entry, Entry):
            raise TypeError()

        # Warn if overwriting an previous citation
        new_citation = entry.to_string("bibtex")
        if key in self._all_citations and new_citation != self._all_citations[key]:
            warnings.warn(f"Replacing citation for {key}")

        # Add to database
        self._all_citations[key] = new_citation

    def _add_citation_tag(self, key, entry):
        """Adds a tag for a citation key which represents the name of the class that
        called :meth:`register`"""

        # Add a citation tag to the citation_tags ordered dictionary with
        # the key being the citation itself and the value being the name of the class
        self._citation_tags[key] = entry

    @property
    def _cited(self):
        """Return a list of the BibTeX entries that have been cited"""
        return [self._all_citations[key] for key in self._papers_to_cite]

    def register(self, key):
        """Register a paper to be cited. The intended use is that :meth:`register`
        should be called only when the referenced functionality is actually being used.

        .. warning::
            Registering a BibTeX citation, with the same key as an existing citation,
            will overwrite the current citation.

        Parameters
        ----------
        key : str
            - The citation key for an entry in `pybamm/CITATIONS.txt` or
            - One or more BibTeX formatted citations
        """
        if self._citation_err_msg is None:
            # Check if citation is a known key
            if key in self._all_citations:
                self._papers_to_cite.add(key)
                # Add citation tags for the key
                # This is used for verbose output
                try:
                    caller = Citations._caller_name()
                    self._add_citation_tag(key, entry=caller)
                # Don't add citation tags if the citation is registered manually
                except KeyError:  # pragma: no cover
                    pass
                return

            # Try to parse the citation using pybtex
            try:
                # Parse string as a bibtex citation, and check that a citation was found
                bib_data = parse_string(key, bib_format="bibtex")
                if not bib_data.entries:
                    raise PybtexError("no entries found")

                # Add and register all citations
                for key, entry in bib_data.entries.items():
                    self._add_citation(key, entry)
                    self.register(key)
                    return
            except PybtexError:
                # Unable to parse / unknown key
                raise KeyError(f"Not a bibtex citation or known citation: {key}")

    def tag_citations(self):  # pragma: no cover
        """Prints the citations tags for the citations that have been registered
        (non-manually). This is used for verbose output when printing citations
        such that it can be seen which citations were registered by PyBaMM classes.

        To use, either call :meth:`tag_citations` after calling :meth:`register`
        for all citations, or enable verbose output with :meth:`print_citations`
        or :meth:`print`.

        .. note::
            If a citation is registered manually, it will not be tagged.

        Examples
        --------
        .. code-block:: python
            :linenos:

            pybamm.citations.register("Doyle1993")
            pybamm.citations.print() or pybamm.print_citations()

        will print the following:

        .. code-block::

            Citations registered:
            Sulzer2021 was cited due to the use of
            pybamm.models.full_battery_models.lithium_ion.dfn

        """
        if self._citation_tags:
            print("\n Citations registered: \n")
            for key, entry in self._citation_tags.items():
                print(f"{key} was cited due to the use of {entry}")

    def print(self, filename=None, output_format="text", verbose=False):
        """Print all citations that were used for running simulations. The verbose
        option is provided to print the citation tags for the citations that have
        been registered non-manually. This is available only upon printing
        to the terminal.

        Parameters
        ----------
        filename : str, optional
            Filename to which to print citations. If None, citations are printed
            to the terminal.
        verbose: bool, optional
            If True, prints the citation tags for the citations that have been
            registered
        """
        if output_format == "text":
            citations = pybtex.format_from_strings(
                self._cited, style="plain", output_backend="plaintext"
            )
        elif output_format == "bibtex":
            citations = "\n".join(self._cited)
        else:
            raise pybamm.OptionError(
                "Output format {} not recognised."
                "It should be 'text' or 'bibtex'.".format(output_format)
            )

        if filename is None:
            print(citations)
            if verbose:
                self.tag_citations()  # pragma: no cover
        else:
            with open(filename, "w") as f:
                f.write(citations)


def print_citations(filename=None, output_format="text", verbose=False):
    """See :meth:`Citations.print`"""
    if citations._citation_err_msg is not None:
        raise ImportError(
            f"Citations could not be registered. If you are on Google Colab - "
            "pybtex does not work with Google Colab due to a known bug - "
            "https://bitbucket.org/pybtex-devs/pybtex/issues/148/. "
            "Please manually cite all the references."
            "\nError encountered -\n"
            f"{citations._citation_err_msg}"
        )
    else:
        if verbose:  # pragma: no cover
            warnings.warn(
                "Verbose output is not available for printing to files, only to the terminal"  # noqa: E501
            )
            pybamm.citations.print(filename, output_format, verbose=True)
        else:
            pybamm.citations.print(filename, output_format)


citations = Citations()
