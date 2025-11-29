import numbers
from itertools import chain

import numpy as np
from scipy.sparse import csr_matrix

import pybamm


class DiffSLExport:
    def __init__(self, model: pybamm.BaseModel):
        self.model = model

    def get_all_parameters(self) -> list[str]:
        params = self.model.parameters

        # filter out length parameters ending in "[m]"
        params = [p for p in params if not p.name.endswith("[m]")]

        # filter out "Number of *"
        params = [p for p in params if not p.name.startswith("Number of ")]

        # filter out constants
        params = [p for p in params if "constant" not in p.name.lower()]

        # do some manual filtering for stuff we don't support yet
        filter_out = [
            # results in addition of two sparse vectors with different sparcity patterns
            "Ambient temperature [K]",
            # results in addition of two sparse vectors with different sparcity patterns
            "Positive electrode diffusivity [m2.s-1]",
            # results in addition of two sparse vectors with different sparcity patterns
            "Negative electrode diffusivity [m2.s-1]",
            # need to implement max(v)/min(v), where v in a sparse or dense vector
            "Maximum concentration in negative electrode [mol.m-3]",
            # need to implement max(v)/min(v), where v in a sparse or dense vector
            "Maximum concentration in positive electrode [mol.m-3]",
        ]
        params = [p for p in params if p.name not in filter_out]

        return [p.name for p in params]

    def get_all_outputs(self) -> list[str]:
        all_vars = self.model.variables

        vars = []
        for var in all_vars.keys():
            reject = False
            for _s in all_vars[var].pre_order():
                # only scalar outputs
                eval_for_shape = all_vars[var].evaluate_for_shape()
                if not isinstance(eval_for_shape, numbers.Number):
                    shape = eval_for_shape.shape
                    if shape != (1, 1):
                        reject = True

            if not reject:
                vars.append(var)

        # filter out some variables that we don't support yet
        filter_out = [
            # results in ExplicitTimeIntegral
            "Discharge capacity [A.h]",
            # results in ExplicitTimeIntegral
            "Throughput capacity [A.h]",
            # need to implement min(v), where v in a sparse or dense vector
            "Minimum negative particle concentration",
            # need to implement sign function (map to copysign),
            # and comparison operators <, >, <=, >=
            "Resistance [Ohm]",
        ]
        vars = [v for v in vars if v not in filter_out]

        # filter out all minimum/maximum outputs,
        # we need to implement min(v) and max(v), where v in a sparse or dense vector
        vars = [
            v for v in vars if "minimum" not in v.lower() and "maximum" not in v.lower()
        ]

        return vars

    def to_diffeq(self, inputs: list[str], outputs: list[str]) -> str:
        """Convert a pybamm model to a diffeq model"""
        model = self.model.new_copy()
        params = self.model.default_parameter_values
        params_names = params.keys()
        for inpt in inputs:
            if not isinstance(inpt, str):
                raise TypeError("inputs must be a list of str")
            if inpt not in params_names:
                raise ValueError(f"input {inpt} not in params")
            params[inpt] = "[input]"
        if len(outputs) == 0:
            raise ValueError("outputs must be a non-empty list of str")
        for out in outputs:
            if not isinstance(out, str):
                raise TypeError("outputs must be a list of str")
            if out not in model.variables:
                raise ValueError(f"output {out} not in model")
            eqn = model.variables[out]
            eval_for_shape = eqn.evaluate_for_shape()
            if isinstance(eval_for_shape, numbers.Number):
                shape = (1, 1)
            else:
                shape = eval_for_shape.shape
        sim = pybamm.Simulation(model, parameter_values=params)
        sim.build()
        model = sim._built_model
        has_events = len(model.events) > 0
        is_ode = model.len_alg == 0

        states = list(chain(model.rhs.keys(), model.algebraic.keys()))
        state_labels = [to_variable_name(v.name) for v in states]
        initial_conditions = [model.initial_conditions[v] for v in states]
        diffeq = {}
        new_line = "\n"

        # inputs
        for inpt in inputs:
            lines = [f"{to_variable_name(inpt)} " + "{"]
            lines += ["  1"]
            diffeq[to_variable_name(inpt)] = new_line.join(lines) + new_line + "}"

        # extract constant vectors and matrices from model as tensors
        symbol_to_tensor_name = {}
        tensor_index = 0
        vectors: set[pybamm.Vector] = set()
        matrices: set[pybamm.Matrix] = set()
        for eqn in chain(
            model.rhs.values(),
            model.algebraic.values(),
            [model.variables[output] for output in outputs],
        ):
            for symbol in eqn.pre_order():
                if isinstance(symbol, pybamm.Vector):
                    vectors.add(symbol)
                    tensor_index += 1
                elif isinstance(symbol, pybamm.Matrix):
                    matrices.add(symbol)
                    tensor_index += 1

        tensor_index = 0
        for symbol in vectors:
            if isinstance(symbol.entries, csr_matrix):
                vector = symbol.entries.toarray().flatten()
            elif isinstance(symbol.entries, np.ndarray):
                vector = symbol.entries.flatten()
            else:
                raise TypeError(f"{type(symbol.entries)} not implemented")

            tensor_name = f"constant{tensor_index}"
            tensor_index += 1

            symbol_to_tensor_name[symbol] = tensor_name

            # any segments of the vector that are the same value are combined
            curr_value = vector[0]
            start_index = 0
            lines = [f"{tensor_name}_i " + "{"]
            for i in range(1, vector.size):
                if vector[i] != curr_value:
                    end_index = i
                    lines += [f"  ({start_index}:{end_index}): {curr_value},"]
                    start_index = i
                    curr_value = vector[i]
            end_index = vector.size
            lines += [f"  ({start_index}:{end_index}): {curr_value},"]

            diffeq[tensor_name] = new_line.join(lines) + new_line + "}"

        for symbol in matrices:
            tensor_name = f"constant{tensor_index}"
            tensor_index += 1
            symbol_to_tensor_name[symbol] = tensor_name
            if isinstance(symbol.entries, csr_matrix):
                nrows, ncols = symbol.entries.shape
                lines = [f"{tensor_name}_ij " + "{"]
                is_diagonal = True
                is_constant = True
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
                        if value != symbol.entries.data[0]:
                            is_constant = False
                        if rowi != colj:
                            is_diagonal = False
                if is_diagonal and is_constant:
                    lines += [
                        f"  (0..{nrows - 1},0..{ncols - 1}): {symbol.entries.data[0]},"
                    ]
                else:
                    for rowi in range(nrows):
                        for j in range(
                            symbol.entries.indptr[rowi], symbol.entries.indptr[rowi + 1]
                        ):
                            colj = symbol.entries.indices[j]
                            lines += [f"  ({rowi},{colj}): {symbol.entries.data[j]},"]
                    if max_colj < ncols - 1 or max_rowi < nrows - 1:
                        # add a zero entry to the end to make sure the matrix is the right size
                        lines += [f"  ({nrows - 1},{ncols - 1}): 0.0,"]

            elif isinstance(symbol.entries, np.ndarray):
                lines = [f"{tensor_name}_ij " + "{"]
                nrows, ncols = symbol.entries.shape
                for rowi in range(nrows):
                    for colj in range(ncols):
                        lines += [f"  ({rowi},{colj}): {symbol.entries[rowi, colj]},"]
            else:
                raise TypeError(f"{type(symbol.entries)} not implemented")

            diffeq[tensor_name] = new_line.join(lines) + new_line + "}"

        # state vector u
        input_lines = []
        lines = ["u_i {"]
        start_index = 0
        y_slice_to_label = {}
        for i, ic in enumerate(initial_conditions):
            label = state_labels[i]
            indices = (
                f"({start_index}:{start_index + ic.size}): " if ic.size > 1 else ""
            )
            input_values = (
                str(ic.entries).replace("[", " ").replace("]", "").replace("\n", ",\n")
            )
            input_lines += [f"{label}input_i {{ \n{input_values}\n}}\n"]
            y_slice_to_label[start_index, start_index + ic.size] = label

            lines += [f"  {label} = {label}input_i,"]
            start_index += ic.size
        diffeq["u"] = new_line.join(input_lines) + new_line.join(lines) + new_line + "}"

        # diff of state vector u
        if not is_ode:
            lines = ["dudt_i {"]
            yp_slice_to_label = {}
            start_index = 0
            for i, ic in enumerate(initial_conditions):
                label = state_name_to_dstate_name(state_labels[i])
                indices = (
                    f"({start_index}:{start_index + ic.size}): " if ic.size > 1 else ""
                )
                zero = pybamm.Scalar(0)
                eqn = equation_to_diffeq(zero, start_index, {}, symbol_to_tensor_name)
                lines += [f"  {indices}{label} = {eqn},"]
                yp_slice_to_label[(start_index, start_index + ic.size)] = label
                start_index += ic.size
            diffeq["dudt"] = new_line.join(lines) + new_line + "}"

        # extract matrix * vector products from model as pre-calculated tensors
        matrix_vector_products = []
        for eqn in chain(
            model.rhs.values(),
            model.algebraic.values(),
            [model.variables[output] for output in outputs],
        ):
            # skip top-level matrix-vector products
            for child in eqn.children:
                for symbol in child.pre_order():
                    if (
                        isinstance(symbol, pybamm.BinaryOperator)
                        and symbol.name == "@"
                        and isinstance(symbol.left, pybamm.Matrix)
                    ):
                        n = symbol.left.entries.shape[1]

                        # check that rhs is a vector with n rows
                        eval_for_shape = symbol.right.evaluate_for_shape()
                        if isinstance(eval_for_shape, numbers.Number):
                            continue
                        shape = eval_for_shape.shape
                        if shape[0] != n:
                            continue

                        # check that rhs has a state vector or dot state vector somewhere
                        has_state_vector = False
                        for s in symbol.right.pre_order():
                            if isinstance(
                                s, pybamm.StateVector | pybamm.StateVectorDot
                            ):
                                has_state_vector = True
                                break
                        if not has_state_vector:
                            continue
                        matrix_vector_products += [symbol]

        tensor_index = 0
        for symbol in matrix_vector_products:
            if symbol in symbol_to_tensor_name:
                continue
            tensor_name = f"varying{tensor_index}"
            tensor_index += 1
            lines = [f"{tensor_name}_i " + "{"]
            eqn = equation_to_diffeq(
                symbol, start_index, y_slice_to_label, symbol_to_tensor_name
            )
            lines += [f"  {eqn},"]
            symbol_to_tensor_name[symbol] = tensor_name
            diffeq[tensor_name] = new_line.join(lines) + new_line + "}"

        # M
        if not is_ode:
            lines = ["M_i {"]
            start_index = 0
            for i, rhs in enumerate(model.rhs.values()):
                eqn = f"{state_name_to_dstate_name(state_labels[i])}_i"
                lines += [f"  {eqn},"]
                start_index += rhs.size
            for algebraic in model.algebraic.keys():
                eqn = "0.0"
                lines += [f"  {eqn},"]
                start_index += algebraic.size
            diffeq["F"] = new_line.join(lines) + new_line + "}"

        # F
        lines = ["F_i {"]
        for rhs in model.rhs.values():
            eqn = equation_to_diffeq(
                rhs, start_index, y_slice_to_label, symbol_to_tensor_name
            )
            lines += [f"  {eqn},"]
            start_index += rhs.size
        for algebraic in model.algebraic.keys():
            eqn = equation_to_diffeq(
                algebraic, start_index, y_slice_to_label, symbol_to_tensor_name
            )
            lines += [f"  {eqn},"]
            start_index += algebraic.size
        diffeq["F"] = new_line.join(lines) + new_line + "}"

        # out
        lines = ["out_i {"]
        for output in outputs:
            eqn = equation_to_diffeq(
                model.variables[output],
                start_index,
                y_slice_to_label,
                symbol_to_tensor_name,
            )
            lines += [f"  {eqn},"]
        diffeq["out"] = new_line.join(lines) + new_line + "}"

        # events
        if has_events:
            lines = ["stop_i {"]
            for event in model.events:
                print(event)
                if event.event_type == pybamm.EventType.TERMINATION:
                    eqn = equation_to_diffeq(
                        event.expression,
                        start_index,
                        y_slice_to_label,
                        symbol_to_tensor_name,
                    )
                    lines += [f"  {eqn},"]
            diffeq["stop"] = new_line.join(lines) + new_line + "}"

        all_lines = [f"in = [{', '.join([to_variable_name(p) for p in inputs])}]"]

        if is_ode:
            state_tensors = ["u"]
            f_and_g_and_out = ["F", "out"]
        else:
            state_tensors = ["u", "dudt"]
            f_and_g_and_out = ["M", "F", "out"]
        if has_events:
            f_and_g_and_out += ["stop"]

        # inputs and constants
        for inpt in inputs:
            all_lines += [diffeq[to_variable_name(inpt)]]
        for key in diffeq.keys():
            if key.startswith("constant"):
                all_lines += [diffeq[key]]
        for key in state_tensors:
            all_lines += [diffeq[key]]
        for key in diffeq.keys():
            if key.startswith("varying"):
                all_lines += [diffeq[key]]
        for key in f_and_g_and_out:
            all_lines += [diffeq[key]]
        return "\n".join(all_lines)


def _equation_to_diffeq(
    equation: pybamm.Symbol,
    y_slice_to_label: dict[tuple[int], str],
    symbol_to_tensor_name: dict[pybamm.Symbol, str],
    transpose: bool = False,
) -> str:
    if isinstance(equation, pybamm.BinaryOperator):
        left = _equation_to_diffeq(
            equation.left, y_slice_to_label, symbol_to_tensor_name
        )

        if equation.name == "@" and isinstance(equation.left, pybamm.Matrix):
            if equation in symbol_to_tensor_name:
                return f"{symbol_to_tensor_name[equation]}_i"
            else:
                right = _equation_to_diffeq(
                    equation.right, y_slice_to_label, symbol_to_tensor_name, True
                )
                return f"({left} * {right})"

        right = _equation_to_diffeq(
            equation.right, y_slice_to_label, symbol_to_tensor_name
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
        return f"{equation.name}({_equation_to_diffeq(equation.child, y_slice_to_label, symbol_to_tensor_name)})"
    elif isinstance(equation, pybamm.Function):
        name = equation.function.__name__
        args = ", ".join(
            [
                _equation_to_diffeq(x, y_slice_to_label, symbol_to_tensor_name)
                for x in equation.children
            ]
        )
        return f"{name}({args})"
    elif isinstance(equation, pybamm.Scalar):
        return f"{equation.value}"
    elif isinstance(equation, pybamm.Vector):
        all_same = all(equation.entries == equation.entries[0, 0]) and isinstance(
            equation.entries, np.ndarray
        )
        if all_same:
            return f"{equation.entries[0, 0]}"
        if equation not in symbol_to_tensor_name:
            raise ValueError(f"equation {equation} not in symbol_to_tensor_name")
        index = "j" if transpose else "i"
        return f"{symbol_to_tensor_name[equation]}_{index}"
    elif isinstance(equation, pybamm.Matrix):
        if equation not in symbol_to_tensor_name:
            raise ValueError(f"equation {equation} not in symbol_to_tensor_name")
        return f"{symbol_to_tensor_name[equation]}_ij"
    elif isinstance(equation, pybamm.StateVector):
        if len(equation.y_slices) != 1:
            raise NotImplementedError("only one state vector slice supported")
        start = equation.y_slices[0].start
        end = equation.y_slices[0].stop
        if (start, end) not in y_slice_to_label:
            raise ValueError(f"equation {equation} not in slice_to_label")
        label = y_slice_to_label[(start, end)]
        index = "j" if transpose else "i"
        return f"{label}_{index}"
    elif isinstance(equation, pybamm.InputParameter):
        return f"{to_variable_name(equation.name)}"
    elif isinstance(equation, pybamm.Time):
        return "t"
    else:
        raise TypeError(f"{type(equation)} not implemented")


def equation_to_diffeq(
    equation: pybamm.Symbol,
    start_index: int,
    slice_to_label: dict[tuple[int], str],
    symbol_to_tensor_name: dict[pybamm.Symbol, str],
) -> str:
    if not isinstance(equation, pybamm.Symbol):
        raise TypeError("equation must be a pybamm.Symbol")
    return _equation_to_diffeq(equation, slice_to_label, symbol_to_tensor_name)


def to_variable_name(name: str) -> str:
    """Convert a name to a valid diffeq variable name"""
    convert_to_underscore = [" ", "-", "(", ")", "[", "]", "{", "}", "/", "\\", "."]
    name = name.lower()
    for char in convert_to_underscore:
        name = name.replace(char, "")
    return name


def vector_to_diffeq_constant(
    vector: pybamm.Vector,
    start_index: int,
    value: int | None = None,
    label: str | None = None,
) -> str:
    """Convert a vector to a diffeq constant"""
    if isinstance(vector, pybamm.Vector) and isinstance(vector.entries, np.ndarray):
        if value is None:
            value = vector.entries[0, 0]
            all_same = True
        else:
            value = vector.entries[0, 0]
            all_same = all(vector.entries == value)
        n = vector.entries.size
        indices = f"({start_index}:{start_index + n}): " if n > 1 else ""
        label = f"{label} = " if label is not None else ""
        if n == 1 or all_same:
            return f"{indices}{label}{value}"
        else:
            raise NotImplementedError(
                "Cannot convert vector with different entries to diffeq constant"
            )
    else:
        raise TypeError("vector must be a pybamm.Vector and ndarray")


def state_name_to_dstate_name(state_name: str) -> str:
    """Convert a state name to a dstate name"""
    return f"d{state_name}dt"
