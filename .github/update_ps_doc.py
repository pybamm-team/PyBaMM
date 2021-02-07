from collections import defaultdict
import os
import re

import pybtex

import pybamm
from pybamm.parameters import parameter_sets


DOC_INTRO = """
Parameter sets from papers. The 'citation' entry provides a reference to the appropriate
paper in the file "pybamm/CITATIONS.txt". To see which parameter sets have been used in
your simulation, add the line "pybamm.print_citations()" to your script.
"""


def main():
    parameter_set_dict = defaultdict(list)
    for ps_name, ps_dict in parameter_sets.__dict__.items():
        if not isinstance(ps_dict, dict):
            continue
        elif "citation" not in ps_dict or "chemistry" not in ps_dict:
            continue

        chemistry = ps_dict["chemistry"]
        citation = ps_dict["citation"]

        # Enclose citation in a list if not already enclosed
        if not isinstance(citation, list):
            citation = [citation]

        parameter_set_dict[chemistry].append((ps_name, citation))

    output_list = [DOC_INTRO]
    citations_file = os.path.join(pybamm.root_dir(), "pybamm", "CITATIONS.txt")

    for ps_chemistry in sorted(parameter_set_dict.keys()):
        ps_citations = parameter_set_dict[ps_chemistry]
        chem_name = ps_chemistry.capitalize() + " " + "parameter sets"
        output_list.append(chem_name)
        dashes = "-" * len(ps_chemistry) + "-" * 15
        output_list.append(dashes)

        for ps_name, ps_citation in sorted(ps_citations):
            citations = pybtex.format_from_file(
                citations_file,
                style="plain",
                output_backend="plaintext",
                citations=ps_citation,
                nocite=True,
            )
            # Remove citation labels "[3]"
            citations = re.sub(r"(?:^|\n)(\[\d+\]\s)", "", citations)
            # Break line at the first space before 80 characters
            citations = re.findall(r"(.{1,80})(?:\s|$)", citations)
            citations = "\n".join(map(lambda x: " " * 8 + x, citations))
            ps_doc = f"    * {ps_name:} :\n{citations}"
            output_list.append(ps_doc)

    output = "\n".join(output_list)
    output += "\n"

    with open(
        os.path.join(pybamm.root_dir(), "pybamm", "parameters", "parameter_sets.py"),
        "r+",
    ) as ps_fp:
        ps_output = ps_fp.read()
        ps_output = ps_output.replace(parameter_sets.__doc__, output)
        ps_fp.truncate(0)
        ps_fp.seek(0)
        ps_fp.write(ps_output)


if __name__ == "__main__":
    main()
