#
# Latexify class
#
import copy
import operator
import re
import warnings

import sympy

import pybamm
from pybamm.expression_tree.printing.sympy_overrides import custom_print_func


def get_rng_min_max_name(rng, min_or_max):
    if getattr(rng[min_or_max], "print_name", None) is None:
        return rng[min_or_max]
    else:
        return rng[min_or_max].print_name


class Latexify:
    """
    Converts all model equations in latex.

    Parameters
    ----------
    filename: str (optional)
        Accepted file formats - any image format, pdf and tex
        Default is None, When None returns all model equations in latex
        If not None, returns all model equations in given file format.

    newline: bool (optional)
        Default is True, If True, returns every equation in a new line.
        If False, returns the list of all the equations.

    Load model
    >>> model = pybamm.lithium_ion.SPM()

    This will returns all model equations in png
    >>> model.latexify("equations.png")

    This will return all the model equations in latex
    >>> model.latexify()

    This will return the list of all the model equations
    >>> model.latexify(newline=False)

    This will return first five model equations
    >>> model.latexify(newline=False)[1:5]
    """

    def __init__(self, model, filename=None, newline=True):
        self.model = model
        self.filename = filename
        self.newline = newline

    def _get_concat_displays(self, node):
        """
        Returns all the concatenation nodes by doing a depth first search through the
        entire equation tree with ranges in front of all nodes.
        """
        concat_displays = []
        dfs_nodes = [node]
        while dfs_nodes:
            node = dfs_nodes.pop()
            if (
                hasattr(node, "concat_latex")
                and getattr(node, "print_name", None) is not None
            ):
                # Combine list of concatenations with list of ranges
                concat_geo = map(
                    operator.add,
                    node.concat_latex,
                    self._get_concat_geometry_displays(node),
                )

                # Add cases and split by new line
                concat_sym = (
                    r"\begin{cases}" + r" \\ ".join(concat_geo) + r"\end{cases}"
                )
                concat_eqn = sympy.Eq(
                    sympy.Symbol(node.print_name),
                    sympy.Symbol(concat_sym),
                    evaluate=False,
                )
                concat_displays.append(concat_eqn)
            dfs_nodes.extend(node.children)

        # Remove duplicates from the list whilst preserving order
        return list(dict.fromkeys(concat_displays))

    def _get_concat_geometry_displays(self, var):
        """Returns a list of min/max ranges of all concatenation nodes in latex."""
        geo = []

        # Loop through all subdomains for concatenations
        for domain in var.domain:
            for var_name, rng in self.model.default_geometry[domain].items():
                if "min" in rng and "max" in rng:
                    rng_min = get_rng_min_max_name(rng, "min")
                    rng_max = get_rng_min_max_name(rng, "max")

                    name = sympy.latex(var_name)
                    geo_latex = f"& {rng_min} < {name} < {rng_max}"
                    geo.append(geo_latex)

        return geo

    def _get_geometry_displays(self, var):
        """
        Returns min range from the first domain and max range from the last domain of
        all nodes in latex.
        """
        geo = []

        if not var.domain:
            return geo

        rng_min = None
        rng_max = None
        name = None

        # Take range minimum from the first domain
        for var_name, rng in self.model.default_geometry[var.domain[0]].items():
            # Trim name (r_n --> r)
            name = re.findall(r"(.)_*.*", str(var_name))[0]
            rng_min = get_rng_min_max_name(rng, "min")

        # Take range maximum from the last domain
        for var_name, rng in self.model.default_geometry[var.domain[-1]].items():
            rng_max = get_rng_min_max_name(rng, "max")

        geo_latex = f"\quad {rng_min} < {name} < {rng_max}"
        geo.append(geo_latex)

        return geo

    def _get_bcs_displays(self, lhs_dr, var):
        """
        Returns a list of boundary condition equations with ranges in front of
        the equations.
        """
        bcs_eqn_list = []
        bcs = self.model.boundary_conditions.get(var, None)

        if bcs:
            # Take range minimum from the first domain
            for var_name, rng in self.model.default_geometry[var.domain[0]].items():
                # Trim name (r_n --> r)
                name = re.findall(r"(.)_*.*", str(var_name))[0]
                rng_min = get_rng_min_max_name(rng, "min")

                bcs_left = sympy.latex(bcs["left"][0].to_equation())
                bcs_left_latex = bcs_left + f"\quad {name} = {rng_min}"
                bcs_eqn = sympy.Eq(lhs_dr, sympy.Symbol(bcs_left_latex), evaluate=False)
                bcs_eqn_list.append(bcs_eqn)

            # Take range maximum from the last domain
            for var_name, rng in self.model.default_geometry[var.domain[-1]].items():
                # Trim name (r_n --> r)
                name = re.findall(r"(.)_*.*", str(var_name))[0]
                rng_max = get_rng_min_max_name(rng, "max")

                bcs_right = sympy.latex(bcs["right"][0].to_equation())
                bcs_right_latex = bcs_right + f"\quad {name} = {rng_max}"
                bcs_eqn = sympy.Eq(
                    lhs_dr, sympy.Symbol(bcs_right_latex), evaluate=False
                )
                bcs_eqn_list.append(bcs_eqn)

        return bcs_eqn_list

    def _get_param_var(self, node):
        """Returns a list of parameters and a list of variables."""
        param_list = []
        var_list = []
        dfs_nodes = [node]

        while dfs_nodes:
            node = dfs_nodes.pop()
            if getattr(node, "print_name", None) is not None:
                # Make a copy of node
                node_copy = copy.copy(node)
                node_copy.print_name = None

                # Add spaces between words
                node_copy_eqn = node_copy.to_equation()
                # Typical current [A] --> \text{Typical current [A]}
                if re.search(r"(^[0-9a-zA-Z-\s.-\[\]()]*$)", str(node_copy_eqn)):
                    node_copy_latex = r"\text{" + str(node_copy_eqn) + "}"
                else:
                    node_copy_latex = sympy.latex(node_copy_eqn)

                # lhs = rhs
                node_latex = sympy.Eq(
                    sympy.Symbol(node.print_name), sympy.Symbol(node_copy_latex)
                )
                # If it contains name, append it to var_list
                if isinstance(
                    node_copy,
                    (
                        pybamm.Parameter,
                        pybamm.Variable,
                        pybamm.FunctionParameter,
                        pybamm.Scalar,
                    ),
                ):
                    var_list.append(node_latex)
                # Else append parameters to param_list
                else:
                    param_list.append(node_latex)
            dfs_nodes.extend(node.children)

        return param_list, var_list

    def latexify(self):
        eqn_list = []
        param_list = []
        var_list = []

        # Add model name to the list
        eqn_list.append(
            sympy.Symbol(
                r"\large{\underline{\textbf{" + self.model.name + " Equations}}}"
            )
        )

        for eqn_type in ["rhs", "algebraic"]:
            for var, eqn in getattr(self.model, eqn_type).items():
                var_symbol = sympy.Symbol(var.print_name)

                # Add equation name to the list
                eqn_list.append(sympy.Symbol(r"\\ \textbf{" + str(var) + "}"))

                # Set lhs derivative
                lhs = sympy.Derivative(var_symbol, "t")
                lhs_dr = sympy.Derivative(var_symbol, "r")

                # Override lhs for algebraic
                if eqn_type == "algebraic":
                    lhs = 0

                # Override derivative to partial derivative
                if len(var.domain) != 0 and var.domain != "current collector":
                    lhs_dr.force_partial = True

                    if not eqn_type == "algebraic":
                        lhs.force_partial = True

                # Boundary conditions equations
                bcs = self._get_bcs_displays(lhs_dr, var)

                # Add ranges from geometry in rhs
                geo = self._get_geometry_displays(var)
                if geo:
                    rhs = sympy.latex(sympy.nsimplify(eqn.to_equation()))
                    rhs = sympy.Symbol(rhs + ",".join(geo))
                else:
                    rhs = sympy.nsimplify(eqn.to_equation())

                # Initial conditions equations
                if not eqn_type == "algebraic":
                    init = self.model.initial_conditions.get(var, None)
                    init_eqn = sympy.Eq(var_symbol, init.to_equation(), evaluate=False)
                    init_eqn = sympy.Symbol(sympy.latex(init_eqn) + r"\quad at\; t=0")

                # Make equation from lhs and rhs
                lhs_rhs = sympy.Eq(lhs, rhs, evaluate=False)

                # Get all concatenation nodes
                concat_displays = self._get_concat_displays(eqn)

                # Set SymPy's init printing to use CustomPrint from sympy_overrides.py
                sympy.init_printing(
                    use_latex=True,
                    latex_mode="plain",
                    latex_printer=custom_print_func,
                    use_unicode="True",
                )

                # Add model equations to the list
                eqn_list.append(lhs_rhs)

                # Add concatenation to the list
                if concat_displays:
                    eqn_list.append(concat_displays)

                # Add initial conditions to the list
                if not eqn_type == "algebraic":
                    eqn_list.extend([init_eqn])

                # Add boundary condition equations to the list
                eqn_list.extend(bcs)

                # Add parameters and variables to the list
                list1, list2 = self._get_param_var(eqn)
                param_list.extend(list1)
                var_list.extend(list2)

        # Add voltage expression to the list
        if "Voltage [V]" in self.model.variables:
            voltage = self.model.variables["Voltage [V]"].to_equation()
            voltage_eqn = sympy.Eq(sympy.Symbol("V"), voltage, evaluate=False)
            # Add voltage to the list
            eqn_list.append(sympy.Symbol(r"\\ \textbf{Voltage [V]}"))
            eqn_list.extend([voltage_eqn])

        # Remove duplicates from the list whilst preserving order
        param_list = list(dict.fromkeys(param_list))
        var_list = list(dict.fromkeys(var_list))
        # Add Parameters and Variables to the list
        eqn_list.append(sympy.Symbol(r"\\ \textbf{Parameters and Variables}"))
        # Add parameters to the list
        eqn_list.extend(param_list)
        # Add names to the list
        eqn_list.extend(var_list)

        # Split list with new lines
        eqn_new_line = sympy.Symbol(r"\\\\".join(map(custom_print_func, eqn_list)))

        # Return latex of equations
        if self.filename is None:
            if self.newline is True:
                return eqn_new_line
            else:
                return eqn_list

        # # Formats - tex
        elif self.filename.endswith(".tex"):  # pragma: no cover
            return sympy.preview(eqn_new_line, outputTexFile=self.filename)

        elif self.filename is not None:
            # Formats - pdf
            if self.filename.endswith(".pdf"):
                return sympy.preview(
                    eqn_new_line,
                    output="pdf",
                    viewer="file",
                    filename=self.filename,
                    euler=False,
                )

            # For more dvioptions see https://www.nongnu.org/dvipng/dvipng_4.html
            else:
                try:
                    return sympy.preview(
                        eqn_new_line,
                        viewer="file",
                        filename=self.filename,
                        dvioptions=["-D", "900"],
                        euler=False,
                    )

                # When equations are too huge, set output resolution to default
                except RuntimeError:  # pragma: no cover
                    warnings.warn(
                        "RuntimeError - Setting the output resolution to default"
                    )
                    return sympy.preview(
                        eqn_new_line,
                        viewer="file",
                        filename=self.filename,
                        euler=False,
                    )
