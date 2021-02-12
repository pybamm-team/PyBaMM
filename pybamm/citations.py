#
# Bibliographical information for PyBaMM
# Inspired by firedrake/PETSc citation workflow
# https://www.firedrakeproject.org/citing.html
#
import pybamm
import os
import pybtex


class Citations:

    """Entry point to citations management.
    This object may be used to record Bibtex citation information and then register that
    a particular citation is relevant for a particular simulation. For a list of all
    possible citations, see `pybamm/CITATIONS.txt`

    Examples
    --------
    >>> import pybamm
    >>> pybamm.citations.register("Sulzer2020")
    >>> pybamm.print_citations("citations.txt")
    """

    def __init__(self):
        self.read_citations()
        self._reset()

    def _reset(self):
        """Reset citations to default only (only for testing purposes)"""
        # Initialize empty papers to cite
        self._papers_to_cite = set()
        # Register the PyBaMM paper and the numpy paper
        self.register("Sulzer2020")
        self.register("Harris2020")

    def read_citations(self):
        """Read the citations text file"""
        self._all_citations = {}

        citations_file = os.path.join(pybamm.root_dir(), "pybamm", "CITATIONS.txt")
        citation = ""
        start = True

        for line in open(citations_file):
            # if start is true, we need to find the key
            if start is True:
                # match everything between { and , in the first line to get the key
                brace_idx = line.find("{")
                comma_idx = line.find(",")
                key = line[brace_idx + 1 : comma_idx]
                # turn off start as we now have the right key
                start = False
            citation += line
            # blank line means next block, add citation to dictionary and
            # reset everything
            if line == "\n":
                self._all_citations[key] = citation
                citation = ""
                start = True

        # add the final citation
        self._all_citations[key] = citation

    def register(self, key):
        """Register a paper to be cited. The intended use is that :meth:`register`
        should be called only when the referenced functionality is actually being used.

        Parameters
        ----------
        key : str
            The key for the paper to be cited
        """
        if key not in self._all_citations:
            raise KeyError("'{}' is not a known citation".format(key))
        self._papers_to_cite.add(key)

    def print(self, filename=None, output_format="text"):
        """Print all citations that were used for running simulations.

        Parameters
        ----------
        filename : str, optional
            Filename to which to print citations. If None, citations are printed to the
            terminal.
        """
        citations = ""
        citations_file = os.path.join(pybamm.root_dir(), "pybamm", "CITATIONS.txt")
        if output_format == "text":
            citations = pybtex.format_from_file(
                citations_file,
                "plain",
                citations=self._papers_to_cite,
                output_backend="plaintext",
            )
        elif output_format == "bibtex":
            for key in self._papers_to_cite:
                citations += self._all_citations[key] + "\n"
        else:
            raise pybamm.OptionError(
                "Output format {} not recognised."
                "It should be 'text' or 'bibtex'.".format(output_format)
            )

        if filename is None:
            print(citations)
        else:
            with open(filename, "w") as f:
                f.write(citations)


def print_citations(filename=None, output_format="text"):
    """ See :meth:`Citations.print` """
    pybamm.citations.print(filename, output_format)


citations = Citations()
