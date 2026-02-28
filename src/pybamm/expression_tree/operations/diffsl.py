from itertools import chain

import numpy as np
from scipy.sparse import csr_matrix

import pybamm


class DiffSLExport:
    """
    A class to export parameterised and discretised PyBaMM models to the DiffSL format.

    Attributes:
        model (pybamm.BaseModel): The PyBaMM model to be exported. This model should be
            parameterised and discretised before exporting.
        float_precision (int): The number of significant digits for float representation.
    """

    def __init__(self, model: pybamm.BaseModel, float_precision: int = 20):
        """
        Initializes the DiffSLExport class.

        Args:
            model (pybamm.BaseModel): The model to export.
            float_precision (int): The number of significant digits for float representation.

        Raises:
            ValueError: If float_precision is not a positive integer.
        """
        self.model = model
        if not isinstance(float_precision, int) or float_precision <= 0:
            raise ValueError("float_precision must be a positive integer")
        self.float_precision = float_precision

    @staticmethod
    def _name_tensor(
        symbol: pybamm.Symbol,
        tensor_index: int,
        is_variable: bool = False,
        is_event: bool = False,
    ) -> str:
        if is_variable:
            tensor_name = f"variable{tensor_index}"
        elif is_event:
            tensor_name = f"event{tensor_index}"
        else:
            tensor_name = f"varying{tensor_index}"
        return tensor_name

    @staticmethod
    def scalar_to_diffeq(
        symbol: pybamm.Scalar, tensor_index: int, float_precision: int = 20
    ) -> tuple[str, str]:
        tensor_name = f"constant{tensor_index}"
        tensor_def = (
            f"{tensor_name}_i "
            + "{"
            + f"\n  (0:1): {symbol.value:.{float_precision}g},\n"
            + "}"
        )
        return tensor_name, tensor_def

    @staticmethod
    def vector_to_diffeq(
        symbol: pybamm.Vector, tensor_index: int, float_precision: int = 20
    ) -> tuple[str, str]:
        if isinstance(symbol.entries, csr_matrix):
            vector = symbol.entries.toarray().flatten()
        elif isinstance(symbol.entries, np.ndarray):
            vector = symbol.entries.flatten()
        else:
            raise TypeError(
                f"{type(symbol.entries)} not implemented"
            )  # pragma: no cover

        tensor_name = f"constant{tensor_index}"

        # any segments of the vector that are the same value are combined
        curr_value = vector[0]
        start_index = 0
        lines = [f"{tensor_name}_i " + "{"]
        for i in range(1, vector.size):
            if vector[i] != curr_value:
                end_index = i
                lines += [
                    f"  ({start_index}:{end_index}): {curr_value:.{float_precision}g},"
                ]
                start_index = i
                curr_value = vector[i]
        end_index = vector.size
        lines += [f"  ({start_index}:{end_index}): {curr_value:.{float_precision}g},"]
        new_line = "\n"
        return tensor_name, new_line.join(lines) + new_line + "}"

    @staticmethod
    def matrix_to_diffeq(
        symbol: pybamm.Matrix, tensor_index: int, float_precision: int = 20
    ) -> tuple[str, str]:
        tensor_name = f"constant{tensor_index}"
        if isinstance(symbol.entries, csr_matrix):
            nrows, ncols = symbol.entries.shape
            lines = [f"{tensor_name}_ij " + "{"]
            is_diagonal = nrows == ncols
            is_constant = True
            is_row_vector = nrows == 1
            max_colj = 0
            max_rowi = 0
            for rowi in range(nrows):
                for j in range(
                    symbol.entries.indptr[rowi], symbol.entries.indptr[rowi + 1]
                ):
                    colj = symbol.entries.indices[j]
                    value = symbol.entries.data[j]
                    max_colj = max(max_colj, colj)
                    max_rowi = max(max_rowi, rowi)
                    if abs(value - symbol.entries.data[0]) > 1e-12:
                        is_constant = False
                    if rowi != colj:
                        is_diagonal = False
            if is_diagonal and is_constant and symbol.entries.nnz == min(nrows, ncols):
                min_dim = min(nrows, ncols)
                lines += [
                    f"  (0..{min_dim - 1},0..{min_dim - 1}): {symbol.entries.data[0]:.{float_precision}g},"
                ]
            elif is_row_vector and is_constant and symbol.entries.nnz == ncols:
                lines += [
                    f"  (0,0:{ncols}): {symbol.entries.data[0]:.{float_precision}g},"
                ]
            else:
                for rowi in range(nrows):
                    for j in reversed(
                        range(
                            symbol.entries.indptr[rowi], symbol.entries.indptr[rowi + 1]
                        )
                    ):
                        colj = symbol.entries.indices[j]
                        lines += [
                            f"  ({rowi},{colj}): {symbol.entries.data[j]:.{float_precision}g},"
                        ]

                if max_colj < ncols - 1 or max_rowi < nrows - 1:
                    # add a zero entry to the end to make sure the matrix is the right size
                    lines += [f"  ({nrows - 1},{ncols - 1}): 0.0,"]

        elif isinstance(symbol.entries, np.ndarray):
            nrows, ncols = symbol.entries.shape
            lines = [f"{tensor_name}_ij " + "{"]
            if nrows == 1:
                # any segments of the row vector that are the same value are combined
                curr_value = symbol.entries[0, 0]
                start_index = 0
                for coli in range(ncols):
                    value = symbol.entries[0, coli]
                    if value != curr_value:
                        end_index = coli
                        lines += [
                            f"  (0, {start_index}:{end_index}): {curr_value:.{float_precision}g},"
                        ]
                        start_index = coli
                        curr_value = value
                end_index = ncols
                lines += [
                    f"  (0, {start_index}:{end_index}): {curr_value:.{float_precision}g},"
                ]
            else:
                for rowi in range(nrows):
                    for colj in range(ncols):
                        lines += [
                            f"  ({rowi},{colj}): {symbol.entries[rowi, colj]:.{float_precision}g},"
                        ]
        else:
            raise TypeError(
                f"{type(symbol.entries)} not implemented"
            )  # pragma: no cover
        new_line = "\n"
        return tensor_name, new_line.join(lines) + new_line + "}"

    def extract_pre_calculated_tensors(
        self,
        eqn: pybamm.Symbol,
        symbol_to_tensor_name: dict[pybamm.Symbol, str],
        tensor_index: int,
        y_slice_to_label: dict[tuple[int], str],
        diffeq: dict[str, str],
        symbol_counts: dict[pybamm.Symbol, int],
        is_variable: bool = False,
        is_event: bool = False,
    ) -> int:
        new_line = "\n"
        for symbol in eqn.post_order():
            # extract any binary operators that occur more than twice and dont involve scalars
            if (
                isinstance(
                    symbol,
                    pybamm.BinaryOperator | pybamm.UnaryOperator | pybamm.Function,
                )
                and symbol_counts.get(symbol, 0) > 1
            ):
                if symbol in symbol_to_tensor_name:
                    continue
                has_scalar = any(
                    [isinstance(child, pybamm.Scalar) for child in symbol.children]
                )
                if has_scalar:
                    continue
                tensor_name = DiffSLExport._name_tensor(
                    symbol, tensor_index, is_variable, is_event
                )
                tensor_index += 1
                lines = [f"{tensor_name}_i " + "{"]
                eqn_str = equation_to_diffeq(
                    symbol,
                    y_slice_to_label,
                    symbol_to_tensor_name,
                    float_precision=self.float_precision,
                )
                lines += [f"  {eqn_str},"]
                symbol_to_tensor_name[symbol] = tensor_name
                diffeq[tensor_name] = new_line.join(lines) + new_line + "}"
        # skip top-level
        for child in eqn.children:
            for symbol in child.post_order():
                if (
                    isinstance(symbol, pybamm.BinaryOperator)
                    and symbol.name == "@"
                    and isinstance(symbol.left, pybamm.Matrix)
                ):
                    # extract the matrix vector product as a separate tensor
                    if symbol in symbol_to_tensor_name:
                        continue
                    tensor_name = DiffSLExport._name_tensor(
                        symbol, tensor_index, is_variable, is_event
                    )
                    tensor_index += 1
                    lines = [f"{tensor_name}_i " + "{"]
                    eqn_str = equation_to_diffeq(
                        symbol,
                        y_slice_to_label,
                        symbol_to_tensor_name,
                        float_precision=self.float_precision,
                    )
                    lines += [f"  {eqn_str},"]
                    symbol_to_tensor_name[symbol] = tensor_name
                    diffeq[tensor_name] = new_line.join(lines) + new_line + "}"

                elif isinstance(symbol, pybamm.DomainConcatenation):
                    if symbol in symbol_to_tensor_name:
                        continue
                    tensor_name = DiffSLExport._name_tensor(
                        symbol, tensor_index, is_variable, is_event
                    )
                    tensor_index += 1
                    lines = [f"{tensor_name}_i " + "{"]
                    for child, slices in zip(
                        symbol.children, symbol._children_slices, strict=False
                    ):
                        eqn_str = equation_to_diffeq(
                            child,
                            y_slice_to_label,
                            symbol_to_tensor_name,
                            float_precision=self.float_precision,
                        )
                        for child_dom, child_slice in slices.items():
                            for i, _slice in enumerate(child_slice):
                                s = symbol._slices[child_dom][i]
                                lines += [f"  ({s.start}:{s.stop}): {eqn_str},"]
                    symbol_to_tensor_name[symbol] = tensor_name
                    diffeq[tensor_name] = new_line.join(lines) + new_line + "}"

        return tensor_index

    def to_diffeq(self, outputs: list[str]) -> str:
        """Convert a pybamm model to a diffeq model"""
        model = self.model.new_copy()
        all_vars = model.get_processed_variables_dict()
        if len(all_vars) == 0 and len(model.variables) > 0:
            all_vars = model.variables
        if not isinstance(outputs, list) or any(
            not isinstance(o, str) for o in outputs
        ):
            raise TypeError("outputs must be a list of str")
        if len(outputs) == 0:
            raise ValueError("outputs must be a non-empty list of str")
        for out in outputs:
            if out not in all_vars:
                raise ValueError(f"output {out} not in model")
        has_events = len(model.events) > 0
        is_ode = len(model.algebraic) == 0

        states = list(chain(model.rhs.keys(), model.algebraic.keys()))
        state_labels = [to_variable_name(v.name) for v in states]
        initial_conditions = [model.initial_conditions[v] for v in states]
        diffeq = {}
        new_line = "\n"

        # inputs, find all pybamm.InputParameters in the model
        inputs = []
        for eqn in chain(
            model.rhs.values(),
            model.algebraic.values(),
            [all_vars[output] for output in outputs],
            [e.expression for e in model.events],
        ):
            for symbol in eqn.pre_order():
                if isinstance(symbol, pybamm.InputParameter):
                    input_name = to_variable_name(symbol.name)
                    if input_name not in inputs:
                        inputs.append(input_name)

        if len(inputs) > 0:
            lines = ["in_i {"]
            for inpt in inputs:
                lines += [f"  {to_variable_name(inpt)} = 1,"]
            diffeq["in"] = new_line.join(lines) + new_line + "}"

        # extract constant vectors and matrices from model as tensors
        symbol_to_tensor_name = {}
        vectors = {}
        matrices = {}
        termination_events = [
            e.expression
            for e in model.events
            if e.event_type == pybamm.EventType.TERMINATION
        ]
        for eqn in chain(
            model.rhs.values(),
            model.algebraic.values(),
            [all_vars[output] for output in outputs],
            termination_events,
        ):
            for symbol in eqn.pre_order():
                if isinstance(symbol, pybamm.Vector):
                    vectors[symbol] = None
                elif isinstance(symbol, pybamm.Matrix):
                    matrices[symbol] = None

        tensor_index = 0
        for symbol in vectors.keys():
            tensor_name, tensor_def = DiffSLExport.vector_to_diffeq(
                symbol, tensor_index, float_precision=self.float_precision
            )
            tensor_index += 1
            symbol_to_tensor_name[symbol] = tensor_name
            diffeq[tensor_name] = tensor_def

        for symbol in matrices.keys():
            tensor_name, tensor_def = DiffSLExport.matrix_to_diffeq(
                symbol, tensor_index, float_precision=self.float_precision
            )
            tensor_index += 1
            symbol_to_tensor_name[symbol] = tensor_name
            diffeq[tensor_name] = tensor_def

        # state vector u
        input_lines = []
        lines = ["u_i {"]
        start_index = 0
        y_slice_to_label = {}
        new_line = "\n"
        for i, ic in enumerate(initial_conditions):
            if isinstance(ic, pybamm.Vector):
                tensor_name, tensor_def = DiffSLExport.vector_to_diffeq(
                    ic, tensor_index, float_precision=self.float_precision
                )
            elif isinstance(ic, pybamm.Scalar):
                tensor_name, tensor_def = DiffSLExport.scalar_to_diffeq(
                    ic, tensor_index, float_precision=self.float_precision
                )
            else:
                raise TypeError(
                    f"Initial condition of type {type(ic)} not supported"
                )  # pragma: no cover
            input_lines += [tensor_def]
            tensor_index += 1
            label = state_labels[i]
            y_slice_to_label[(start_index, start_index + ic.size)] = label
            lines += [f"  {label} = {tensor_name}_i,"]
            start_index += ic.size
        diffeq["u"] = (
            new_line.join(input_lines)
            + new_line
            + new_line.join(lines)
            + new_line
            + "}"
        )

        # diff of state vector u
        if not is_ode:
            lines = ["dudt_i {"]
            start_index = 0
            for i, ic in enumerate(initial_conditions):
                label = state_name_to_dstate_name(state_labels[i])
                indices = (
                    f"({start_index}:{start_index + ic.size}): " if ic.size > 1 else ""
                )
                zero = pybamm.Scalar(0)
                eqn = equation_to_diffeq(
                    zero,
                    {},
                    symbol_to_tensor_name,
                    float_precision=self.float_precision,
                )
                lines += [f"  {indices}{label} = {eqn},"]
                start_index += ic.size
            diffeq["dudt"] = new_line.join(lines) + new_line + "}"

        symbol_counts = {}
        for eqn in chain(
            model.rhs.values(),
            model.algebraic.values(),
            termination_events,
            [all_vars[output] for output in outputs],
        ):
            for s in eqn.pre_order():
                if s in symbol_counts:
                    symbol_counts[s] += 1
                else:
                    symbol_counts[s] = 1

        for eqn in chain(
            model.rhs.values(),
            model.algebraic.values(),
        ):
            tensor_index = self.extract_pre_calculated_tensors(
                eqn,
                symbol_to_tensor_name,
                tensor_index,
                y_slice_to_label,
                diffeq,
                symbol_counts,
            )

        for eqn in chain(
            termination_events,
        ):
            tensor_index = self.extract_pre_calculated_tensors(
                eqn,
                symbol_to_tensor_name,
                tensor_index,
                y_slice_to_label,
                diffeq,
                symbol_counts,
                is_event=True,
            )

        for eqn in chain(
            [all_vars[output] for output in outputs],
        ):
            tensor_index = self.extract_pre_calculated_tensors(
                eqn,
                symbol_to_tensor_name,
                tensor_index,
                y_slice_to_label,
                diffeq,
                symbol_counts,
                is_variable=True,
            )

        # M
        if not is_ode:
            lines = ["M_i {"]
            start_index = 0
            for i, rhs in enumerate(model.rhs.values()):
                eqn = f"{state_name_to_dstate_name(state_labels[i])}_i"
                lines += [f"  {eqn},"]
                start_index += rhs.size
            for algebraic in model.algebraic.values():
                n = algebraic.size
                if n == 1:
                    eqn = "0.0"
                else:
                    eqn = f"({start_index}:{start_index + n}): 0.0"
                lines += [f"  {eqn},"]
                start_index += algebraic.size
            diffeq["M"] = new_line.join(lines) + new_line + "}"

        # F
        lines = ["F_i {"]
        for rhs in model.rhs.values():
            eqn = equation_to_diffeq(
                rhs,
                y_slice_to_label,
                symbol_to_tensor_name,
                float_precision=self.float_precision,
            )
            lines += [f"  {eqn},"]
            start_index += rhs.size
        for algebraic in model.algebraic.values():
            eqn = equation_to_diffeq(
                algebraic,
                y_slice_to_label,
                symbol_to_tensor_name,
                float_precision=self.float_precision,
            )
            lines += [f"  {eqn},"]
            start_index += algebraic.size
        diffeq["F"] = new_line.join(lines) + new_line + "}"

        # out
        lines = ["out_i {"]
        for output in outputs:
            eqn = equation_to_diffeq(
                all_vars[output],
                y_slice_to_label,
                symbol_to_tensor_name,
                float_precision=self.float_precision,
            )
            lines += [f"  {eqn},"]
        diffeq["out"] = new_line.join(lines) + new_line + "}"

        # events
        if has_events:
            lines = ["stop_i {"]
            for event in model.events:
                if event.event_type == pybamm.EventType.TERMINATION:
                    eqn = equation_to_diffeq(
                        event.expression,
                        y_slice_to_label,
                        symbol_to_tensor_name,
                        float_precision=self.float_precision,
                    )
                    lines += [f"  {eqn},"]
            diffeq["stop"] = new_line.join(lines) + new_line + "}"

        all_lines = []

        if is_ode:
            state_tensors = ["u"]
            f_and_g = ["F"]
        else:
            state_tensors = ["u", "dudt"]
            f_and_g = ["M", "F"]
        if has_events:
            stop = ["stop"]
        else:
            stop = []
        out = ["out"]

        # inputs and constants
        if "in" in diffeq:
            all_lines = [diffeq["in"]]
        for key in diffeq.keys():
            if key.startswith("constant"):
                all_lines += [diffeq[key]]
        for key in state_tensors:
            all_lines += [diffeq[key]]
        for key in diffeq.keys():
            if key.startswith("varying"):
                all_lines += [diffeq[key]]
        for key in f_and_g:
            all_lines += [diffeq[key]]
        for key in diffeq.keys():
            if key.startswith("event"):
                all_lines += [diffeq[key]]
        for key in stop:
            all_lines += [diffeq[key]]
        for key in diffeq.keys():
            if key.startswith("variable"):
                all_lines += [diffeq[key]]
        for key in out:
            all_lines += [diffeq[key]]

        return "\n".join(all_lines)


def _equation_to_diffeq(
    equation: pybamm.Symbol,
    y_slice_to_label: dict[tuple[int], str],
    symbol_to_tensor_name: dict[pybamm.Symbol, str],
    float_precision: int = 20,
    transpose: bool = False,
) -> str:
    if equation in symbol_to_tensor_name:
        if isinstance(equation, pybamm.Matrix):
            index = "ji" if transpose else "ij"
        else:
            index = "j" if transpose else "i"
        return f"{symbol_to_tensor_name[equation]}_{index}"
    if isinstance(equation, pybamm.BinaryOperator):
        left = _equation_to_diffeq(
            equation.left,
            y_slice_to_label,
            symbol_to_tensor_name,
            float_precision=float_precision,
            transpose=transpose,
        )

        if equation.name == "@" and isinstance(equation.left, pybamm.Matrix):
            right = _equation_to_diffeq(
                equation.right,
                y_slice_to_label,
                symbol_to_tensor_name,
                float_precision=float_precision,
                transpose=not transpose,
            )
            return f"({left} * {right})"
        else:
            right = _equation_to_diffeq(
                equation.right,
                y_slice_to_label,
                symbol_to_tensor_name,
                float_precision=float_precision,
                transpose=transpose,
            )

        if equation.name == "maximum":
            return f"max({left}, {right})"
        elif equation.name == "minimum":
            return f"min({left}, {right})"
        elif equation.name == "@":
            return f"({left} * {right})"
        elif equation.name == "**":
            return f"pow({left}, {right})"
        return f"({left} {equation.name} {right})"
    elif isinstance(equation, pybamm.UnaryOperator):
        return f"{equation.name}({_equation_to_diffeq(equation.child, y_slice_to_label, symbol_to_tensor_name, float_precision=float_precision, transpose=transpose)})"
    elif isinstance(equation, pybamm.Function):
        name = equation.function.__name__
        args = [
            _equation_to_diffeq(
                x,
                y_slice_to_label,
                symbol_to_tensor_name,
                float_precision=float_precision,
                transpose=transpose,
            )
            for x in equation.children
        ]
        if name == "_reg_power_evaluate":
            # three args: x, a, scale (optional)
            # Approximates |x|^a * sign(x) using:
            #  y = x * (x^2 + delta^2)^((a-1)/2)
            # When scale is set:
            #    y = (x/scale) * ((x/scale)^2 + delta^2)^((a-1)/2) * scale^a
            delta = equation.delta
            x = args[0]
            a = args[1]
            if len(equation.children) == 3:
                x = f"({x} / {args[2]})"
                return f"({x} * pow((pow({x}, 2) + {delta**2:.{float_precision}g}), ({a} - 1) / 2)) * pow({args[2]}, {a})"
            else:
                return f"({x} * pow((pow({x}, 2) + {delta**2:.{float_precision}g}), ({a} - 1) / 2))"
        elif name == "_arcsinh2_evaluate":
            # Two-argument arcsinh function for arcsinh(a/b) that avoids division by zero
            # by adding a small regularisation term to the denominator.
            # Computes arcsinh(a / b_eff) where b_eff = sign(b) * hypot(b, eps), where hypot = sqrt(b^2 + eps^2)
            # Note: the sign(b) function treats sign(0) as 1 for numerical stability.
            a = args[0]
            b = args[1]
            eps2 = equation.eps**2
            return f"arcsinh({a} / (copysign(sqrt(pow({b}, 2) + {eps2:.{float_precision}g}), {b})))"
        else:
            # all the other functions have the same signature as in diffsl
            args = ",".join(args)
            return f"{name}({args})"
    elif isinstance(equation, pybamm.Scalar):
        return f"{equation.value:.{float_precision}g}"
    elif isinstance(equation, pybamm.StateVector):
        if len(equation.y_slices) != 1:
            raise NotImplementedError(
                "only one state vector slice supported"
            )  # pragma: no cover
        start = equation.y_slices[0].start
        end = equation.y_slices[0].stop
        index = "j" if transpose else "i"
        # look for exact match
        for sl, label in y_slice_to_label.items():
            if sl[0] == start and sl[1] == end:
                return f"{label}_{index}"
        # didn't find exact match, look for containing slice
        for sl, label in y_slice_to_label.items():
            if sl[0] <= start and sl[1] >= end:
                start_offset = start - sl[0]
                end_offset = end - sl[0]
                return f"{label}_{index}[{start_offset}:{end_offset}]"
        raise ValueError(
            f"equation {equation} not in slice_to_label"
        )  # pragma: no cover

    elif isinstance(equation, pybamm.InputParameter):
        return f"{to_variable_name(equation.name)}"
    elif isinstance(equation, pybamm.Time):
        return "t"
    else:
        raise TypeError(
            f"{type(equation)} not implemented for symbol {equation}"
        )  # pragma: no cover


def equation_to_diffeq(
    equation: pybamm.Symbol,
    slice_to_label: dict[tuple[int], str],
    symbol_to_tensor_name: dict[pybamm.Symbol, str],
    float_precision: int = 20,
    transpose: bool = False,
) -> str:
    if not isinstance(equation, pybamm.Symbol):
        raise TypeError("equation must be a pybamm.Symbol")  # pragma: no cover
    return _equation_to_diffeq(
        equation,
        slice_to_label,
        symbol_to_tensor_name,
        float_precision=float_precision,
        transpose=transpose,
    )


def to_variable_name(name: str) -> str:
    """Convert a name to a valid diffeq variable name"""
    convert_to_underscore = [" ", "-", "(", ")", "[", "]", "{", "}", "/", "\\", "."]
    name = name.lower()
    for char in convert_to_underscore:
        name = name.replace(char, "")
    return name


def state_name_to_dstate_name(state_name: str) -> str:
    """Convert a state name to a dstate name"""
    return f"d{state_name}dt"
