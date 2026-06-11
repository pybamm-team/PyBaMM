"""
Class to regularise sqrt and power operations in a PyBaMM expression tree.
"""

from __future__ import annotations

import pybamm


class RegulariseSqrtAndPower:
    """
    Callable that replaces Sqrt and Power nodes with RegPower nodes.

    All Sqrt and Power nodes are replaced with RegPower. If the base of the
    operation matches a symbol in the scales map, that scale is used;
    otherwise the default scale (None) is used.

    Parameters
    ----------
    scales : dict[pybamm.Symbol, pybamm.Symbol]
        Mapping from original symbols to their scale values.
    inputs : dict[str, pybamm.Symbol]
        The inputs dict from FunctionParameter.
    """

    __slots__ = ["scales", "symbol_to_name"]

    def __init__(
        self,
        scales: dict[pybamm.Symbol, pybamm.Symbol],
        inputs: dict[str, pybamm.Symbol],
    ):
        self.scales = scales
        self.symbol_to_name = {symbol: name for name, symbol in inputs.items()}

    def __call__(
        self,
        symbol: pybamm.Symbol,
        inputs: dict[str, pybamm.Symbol] | None = None,
        **kwargs,
    ) -> pybamm.Symbol:
        """Apply regularisation to an expression tree."""
        if inputs is None:
            inputs = {}

        # Build resolved scales: processed_symbol -> processed_scale
        resolved_scales = {}

        for original_symbol, scale_ref in self.scales.items():
            # Resolve the scale value through inputs if it's a known symbol
            processed_scale = self._rebuild_expr(scale_ref, inputs)
            if processed_scale is None:
                # Keep original if can't resolve
                processed_scale = scale_ref

            # Get the processed symbol
            if original_symbol in self.symbol_to_name:
                name = self.symbol_to_name[original_symbol]
                if name in inputs:
                    processed_symbol = inputs[name]
                    resolved_scales[processed_symbol] = processed_scale
            else:
                # Complex expression - rebuild with processed symbols
                processed_expr = self._rebuild_expr(original_symbol, inputs)
                if processed_expr is not None:
                    resolved_scales[processed_expr] = processed_scale

        return self._process(symbol, resolved_scales)

    def _rebuild_expr(self, expr, inputs):
        """Rebuild an expression using processed symbols."""
        # First check if this exact expression is in symbol_to_name
        # (e.g., maximum(minimum(c_s, c_max), 0.0) is stored as a whole)
        if expr in self.symbol_to_name:
            name = self.symbol_to_name[expr]
            return inputs.get(name)

        # Leaf nodes
        if not expr.children:
            # Keep Scalars, Parameters, and other constants as-is
            # (Parameters will be processed later by parameter substitutor)
            if isinstance(expr, pybamm.Scalar | pybamm.Parameter):
                return expr
            return None

        # Recursive rebuild for expressions with children
        new_children = []
        for child in expr.children:
            rebuilt = self._rebuild_expr(child, inputs)
            if rebuilt is None:
                return None
            new_children.append(rebuilt)

        return expr.create_copy(new_children=new_children)

    def _process(self, sym, resolved_scales):
        """Recursively replace Sqrt/Power with RegPower."""
        if not sym.children:
            return sym

        new_children = [self._process(child, resolved_scales) for child in sym.children]

        if isinstance(sym, pybamm.Sqrt):
            child = new_children[0]
            scale = self._get_scale(child, resolved_scales)
            return pybamm.RegPower(child, 0.5, scale=scale)

        if isinstance(sym, pybamm.Power):
            base, exponent = new_children
            scale = self._get_scale(base, resolved_scales)
            return pybamm.RegPower(base, exponent, scale=scale)

        if any(n is not o for n, o in zip(new_children, sym.children, strict=True)):
            return sym.create_copy(new_children=new_children)
        return sym

    def _get_scale(self, expr, resolved_scales):
        """Get scale for an expression, defaulting to None.

        Only exact matches are used. For patterns like (c_max - c_s),
        they must be explicitly registered in the scales dict.
        """
        for var, scale in resolved_scales.items():
            if expr == var:
                return scale
        return None
