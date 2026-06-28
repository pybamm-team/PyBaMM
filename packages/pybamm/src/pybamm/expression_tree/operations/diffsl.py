from __future__ import annotations

from dataclasses import dataclass
from itertools import chain

import numpy as np
from scipy.sparse import csr_matrix

import pybamm


@dataclass(frozen=True)
class _ExperimentScheduleState:
    schedule_index: int
    stop_expression: pybamm.Symbol
    model_branch_index: int
    target_value: float = 0.0


class DiffSLExport:
    """
    A class to export parameterised and discretised PyBaMM models to the DiffSL format.

    Attributes:
        model (pybamm.BaseModel): The PyBaMM model to be exported. This model should be
            parameterised and discretised before exporting.
        float_precision (int): The number of significant digits for float representation.
    """

    STEP_VALUE_TENSOR_NAME = "constantStepTargetValue"

    def __init__(
        self, model: pybamm.BaseModel | pybamm.Simulation, float_precision: int = 20
    ):
        """
        Initializes the DiffSLExport class.

        Args:
            model (pybamm.BaseModel or pybamm.Simulation): The model or simulation to
                export.
            float_precision (int): The number of significant digits for float representation.

        Raises:
            ValueError: If float_precision is not a positive integer.
        """
        if not isinstance(model, (pybamm.BaseModel, pybamm.Simulation)):
            raise TypeError("model must be a pybamm.BaseModel or pybamm.Simulation")
        self._source = model
        if not isinstance(float_precision, int) or float_precision <= 0:
            raise ValueError("float_precision must be a positive integer")
        self.float_precision = float_precision
        self._schedule_to_model_branch_order = None
        self._input_names = None
        self._nstates = None
        self._preprocess_model()

    def _preprocess_model(self) -> None:
        source = self._source
        self._has_experiment = False

        if isinstance(source, pybamm.Simulation):
            if getattr(source, "experiment", None) is None:
                source.build()
                model = source.built_model
            else:
                source.build_for_experiment()
                if (
                    not source._experiment_uses_unified_model
                    or source._built_experiment_model is None
                ):
                    raise ValueError(
                        "DiffSL export of simulations with experiments requires "
                        "experiment_model_mode='unified'"
                    )
                model = source._built_experiment_model
                self._has_experiment = True
        else:
            model = source

        if model is None:  # pragma: no cover
            raise ValueError("Unable to resolve a built model for DiffSL export")

        self.model = model

        # Cache the base all_vars dict (output-independent portion)
        processed_vars = model.get_processed_variables_dict()
        self._all_vars = processed_vars.copy()
        if len(model.variables) > 0:
            self._all_vars = model.variables.copy() | self._all_vars

        # Cache input names from output-independent equation sources
        self._base_input_names = []
        for eqn in chain(
            model.rhs.values(),
            model.algebraic.values(),
            [e.expression for e in model.events],
        ):
            for symbol in eqn.pre_order():
                if isinstance(symbol, pybamm.InputParameter):
                    if self._has_experiment and symbol.name in (
                        pybamm.Simulation._STEP_INDEX_INPUT,
                        pybamm.Simulation._STEP_VALUE_INPUT,
                    ):
                        continue
                    variable_name = to_variable_name(symbol.name)
                    if variable_name not in [vn for _, vn in self._base_input_names]:
                        self._base_input_names.append((symbol.name, variable_name))

    def _collect_input_names(
        self, all_vars: dict, outputs: list[str]
    ) -> list[tuple[str, str]]:
        """Return ordered (original_name, variable_name) pairs for all InputParameters.

        Starts from the cached output-independent names and appends any additional
        names found only in the given output expressions.
        """
        result = list(self._base_input_names)
        existing_vnames = {vn for _, vn in result}
        for output in outputs:
            for symbol in all_vars[output].pre_order():
                if isinstance(symbol, pybamm.InputParameter):
                    if self._has_experiment and symbol.name in (
                        pybamm.Simulation._STEP_INDEX_INPUT,
                        pybamm.Simulation._STEP_VALUE_INPUT,
                    ):
                        continue  # pragma: no cover
                    variable_name = to_variable_name(symbol.name)
                    if variable_name not in existing_vnames:
                        result.append((symbol.name, variable_name))
                        existing_vnames.add(variable_name)
        return result

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

    @staticmethod
    def _tensor_block(tensor_name: str, entries: list[str], suffix: str = "i") -> str:
        new_line = "\n"
        lines = [f"{tensor_name}_{suffix} " + "{"]
        lines.extend(f"  {entry}," for entry in entries)
        return new_line.join(lines) + new_line + "}"

    def _materialize_expression_tensor(
        self,
        symbol: pybamm.Symbol,
        symbol_to_tensor_name: dict[pybamm.Symbol, str],
        tensor_index: int,
        y_slice_to_label: dict[tuple[int], str],
        diffeq: dict[str, str],
        is_variable: bool = False,
        is_event: bool = False,
    ) -> int:
        if symbol in symbol_to_tensor_name:
            return tensor_index

        tensor_name = DiffSLExport._name_tensor(
            symbol, tensor_index, is_variable, is_event
        )
        tensor_index += 1
        eqn_str = equation_to_diffeq(
            symbol,
            y_slice_to_label,
            symbol_to_tensor_name,
            float_precision=self.float_precision,
            use_model_index=self._has_experiment,
        )
        diffeq[tensor_name] = self._tensor_block(tensor_name, [eqn_str])
        symbol_to_tensor_name[symbol] = tensor_name
        return tensor_index

    def _materialize_conditional_tensor(
        self,
        symbol: pybamm.Conditional,
        symbol_to_tensor_name: dict[pybamm.Symbol, str],
        tensor_index: int,
        y_slice_to_label: dict[tuple[int], str],
        diffeq: dict[str, str],
        is_variable: bool = False,
        is_event: bool = False,
        num_terminal_states: int = 0,
    ) -> int:
        if symbol in symbol_to_tensor_name:
            return tensor_index

        tensor_name = DiffSLExport._name_tensor(
            symbol, tensor_index, is_variable, is_event
        )
        tensor_index += 1

        if symbol.size_for_testing != 1:
            raise NotImplementedError("DiffSL export only supports scalar Conditionals")

        if self._has_experiment and self._schedule_to_model_branch_order is not None:
            branch_order = self._schedule_to_model_branch_order
        else:
            branch_order = list(range(len(symbol.branches)))
            branch_order.append(branch_order[-1])

        entries = [
            equation_to_diffeq(
                symbol.branches[branch_index],
                y_slice_to_label,
                symbol_to_tensor_name,
                float_precision=self.float_precision,
                use_model_index=self._has_experiment,
            )
            for branch_index in branch_order
        ]
        entries.extend(["0.0"] * num_terminal_states)
        diffeq[tensor_name] = self._tensor_block(tensor_name, entries)
        symbol_to_tensor_name[symbol] = tensor_name
        return tensor_index

    @staticmethod
    def _compute_experiment_cycle_length(
        schedule_items: list[object],
    ) -> int:
        """Determine the shortest repeating cycle in the experiment schedule.

        Computes the LPS (longest proper prefix which is also a suffix) array,
        also known as the KMP prefix function or failure table, over the
        sequence of schedule items. If the sequence is composed of
        repetitions of a shorter pattern, returns the length of that pattern.
        Otherwise returns the full sequence length.

        Parameters
        ----------
        schedule_items : list
            Equality-comparable schedule items for every step in the experiment.

        Returns
        -------
        int
            The length of the shortest repeating cycle, or ``len(schedule_items)``
            if no repeating cycle is found.
        """
        n = len(schedule_items)
        if n == 0:
            return 0  # pragma: no cover
        lps = [0] * n
        length = 0
        i = 1
        while i < n:
            if schedule_items[i] == schedule_items[length]:
                length += 1
                lps[i] = length
                i += 1
            elif length != 0:
                length = lps[length - 1]
            else:
                lps[i] = 0
                i += 1
        p = lps[-1]
        if p > 0 and n % (n - p) == 0:
            return n - p
        return n

    @staticmethod
    def _normalise_schedule_value(value):
        if value is None:
            return None
        if isinstance(value, pybamm.Scalar):
            return float(value.value)
        try:
            return float(value)
        except (TypeError, ValueError):
            return repr(value)

    @staticmethod
    def _effective_step_duration(step: pybamm.step.BaseStep, initial_start_time):
        effective_duration = step.duration
        if step.end_time is not None and initial_start_time is not None:
            start_dt = (step.start_time - initial_start_time).total_seconds()
            end_dt = (step.end_time - initial_start_time).total_seconds()
            effective_duration = min(effective_duration, end_dt - start_dt)
        return effective_duration

    @staticmethod
    def _padding_step_duration(
        step: pybamm.step.BaseStep,
        effective_duration: float,
        initial_start_time,
    ):
        if step.next_start_time is None:
            return None
        if initial_start_time is None:  # pragma: no cover
            raise ValueError(
                "Unified experiment DiffSL export expected an initial start time "
                "when step next_start_time is set"
            )
        step_end_rel = (
            step.start_time - initial_start_time
        ).total_seconds() + effective_duration
        next_start_rel = (step.next_start_time - initial_start_time).total_seconds()
        if next_start_rel > step_end_rel:
            return next_start_rel - step_end_rel
        return None

    def _experiment_schedule_key(
        self,
        sim: pybamm.Simulation,
        step: pybamm.step.BaseStep,
        branch_index: int,
    ):
        initial_start_time = sim.experiment.initial_start_time
        effective_duration = self._effective_step_duration(step, initial_start_time)
        padding_duration = self._padding_step_duration(
            step, effective_duration, initial_start_time
        )
        target = step.control_target_value(sim._parameter_values)
        ambient = (
            step.temperature or sim._parameter_values[sim._AMBIENT_TEMPERATURE_INPUT]
        )

        return (
            branch_index,
            self._normalise_schedule_value(effective_duration),
            self._normalise_schedule_value(padding_duration),
            self._normalise_schedule_value(target),
            self._normalise_schedule_value(ambient),
        )

    def _get_unified_experiment_schedule_states(
        self,
        sim: pybamm.Simulation,
        step_branches: list[pybamm.Symbol],
        steptime0_sv: pybamm.StateVector,
    ) -> list[_ExperimentScheduleState]:
        step_indices = sim._experiment_step_indices
        schedule_keys = [
            self._experiment_schedule_key(sim, step, branch_index)
            for step, branch_index in zip(
                sim.experiment.steps, step_indices, strict=True
            )
        ]

        initial_start_time = sim.experiment.initial_start_time
        if initial_start_time is None:
            cycle_length = self._compute_experiment_cycle_length(schedule_keys)
        else:
            cycle_length = len(step_indices)
        cycle_steps = sim.experiment.steps[:cycle_length]
        cycle_indices = step_indices[:cycle_length]

        schedule_states = []

        for step, branch_index in zip(cycle_steps, cycle_indices, strict=True):
            branch = step_branches[branch_index - 1]

            effective_duration = self._effective_step_duration(step, initial_start_time)
            duration_stop = pybamm.Scalar(effective_duration) - steptime0_sv
            if isinstance(branch, pybamm.Scalar) and branch.value == 1:
                stop_expr = duration_stop
            else:
                stop_expr = pybamm.minimum(duration_stop, branch)
            target = step.control_target_value(sim._parameter_values)
            schedule_states.append(
                _ExperimentScheduleState(
                    len(schedule_states),
                    stop_expr,
                    branch_index - 1,
                    target_value=float(target) if target is not None else 0.0,
                )
            )

            padding_duration = self._padding_step_duration(
                step, effective_duration, initial_start_time
            )
            if padding_duration is not None:
                schedule_states.append(
                    _ExperimentScheduleState(
                        len(schedule_states),
                        pybamm.Scalar(padding_duration) - steptime0_sv,
                        sim._experiment_padding_rest_index - 1,
                        target_value=0.0,
                    )
                )

        return schedule_states[-1:] + schedule_states[:-1]

    def _get_stop_expressions(
        self, model: pybamm.BaseModel, steptime0_sv: pybamm.StateVector | None = None
    ) -> tuple[list[pybamm.Symbol], list[int]]:
        general_stops = [
            event.expression
            for event in model.events
            if event.event_type == pybamm.EventType.TERMINATION
            and event.name != pybamm.Simulation._COMBINED_TERMINATION_EVENT
        ]

        if not (self._has_experiment and isinstance(self._source, pybamm.Simulation)):
            return general_stops, []

        sim = self._source
        combined_event = next(
            (
                event
                for event in model.events
                if event.event_type == pybamm.EventType.TERMINATION
                and event.name == pybamm.Simulation._COMBINED_TERMINATION_EVENT
            ),
            None,
        )
        if combined_event is None or not isinstance(
            combined_event.expression, pybamm.Conditional
        ):
            raise ValueError(
                "Unified experiment DiffSL export requires a combined step termination "
                "conditional"
            )  # pragma: no cover

        if steptime0_sv is None:
            raise ValueError(
                "Unified experiment DiffSL export requires a steptime0 state vector"
            )  # pragma: no cover

        step_branches = list(combined_event.expression.branches)
        schedule_states = self._get_unified_experiment_schedule_states(
            sim, step_branches, steptime0_sv
        )
        self._schedule_states = schedule_states
        cycle_stop_exprs = [state.stop_expression for state in schedule_states]
        schedule_stop_order = [state.schedule_index for state in schedule_states]
        schedule_to_model_branch_order = [
            state.model_branch_index
            for state in sorted(schedule_states, key=lambda state: state.schedule_index)
        ]
        self._schedule_to_model_branch_order = schedule_to_model_branch_order
        return cycle_stop_exprs + general_stops, schedule_stop_order

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
        num_terminal_states: int = 0,
    ) -> int:
        for symbol in eqn.post_order():
            if isinstance(symbol, pybamm.Conditional):
                tensor_index = self._materialize_conditional_tensor(
                    symbol,
                    symbol_to_tensor_name,
                    tensor_index,
                    y_slice_to_label,
                    diffeq,
                    is_variable=is_variable,
                    is_event=is_event,
                    num_terminal_states=num_terminal_states,
                )
                continue
            # extract any binary operators that occur more than two times and dont involve scalars
            if (
                isinstance(
                    symbol,
                    pybamm.BinaryOperator | pybamm.UnaryOperator | pybamm.Function,
                )
                and symbol_counts.get(symbol, 0) > 2
            ):
                has_scalar = any(
                    isinstance(child, pybamm.Scalar) for child in symbol.children
                )
                if has_scalar:
                    continue
                tensor_index = self._materialize_expression_tensor(
                    symbol,
                    symbol_to_tensor_name,
                    tensor_index,
                    y_slice_to_label,
                    diffeq,
                    is_variable=is_variable,
                    is_event=is_event,
                )
        # skip top-level
        for child in eqn.children:
            for symbol in child.post_order():
                if (
                    isinstance(symbol, pybamm.BinaryOperator)
                    and symbol.name == "@"
                    and isinstance(symbol.left, pybamm.Matrix)
                ):
                    # extract the matrix vector product as a separate tensor
                    tensor_index = self._materialize_expression_tensor(
                        symbol,
                        symbol_to_tensor_name,
                        tensor_index,
                        y_slice_to_label,
                        diffeq,
                        is_variable=is_variable,
                        is_event=is_event,
                    )

                elif isinstance(symbol, pybamm.DomainConcatenation):
                    if symbol in symbol_to_tensor_name:
                        continue
                    new_line = "\n"
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
                            use_model_index=self._has_experiment,
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
        if not isinstance(outputs, list) or any(
            not isinstance(o, str) for o in outputs
        ):
            raise TypeError("outputs must be a list of str")
        if len(outputs) == 0:
            raise ValueError("outputs must be a non-empty list of str")
        all_vars = self._all_vars.copy()
        for out in outputs:
            if out not in all_vars:
                raise ValueError(f"output {out} not in model")
            if model.symbol_processor:
                try:
                    all_vars[out] = model.get_processed_variable(out)
                except KeyError:  # pragma: no cover
                    pass

        # for experiments we add a variable steptime0 indicating
        # the start time of the current step
        states = list(chain(model.rhs.keys(), model.algebraic.keys()))
        if self._has_experiment:
            total_model_state_size = sum(
                model.initial_conditions[v].size for v in states
            )
            steptime0_slice = slice(total_model_state_size, total_model_state_size + 1)
            steptime0_sv = pybamm.StateVector(steptime0_slice)
        else:
            total_model_state_size = 0
            steptime0_sv = None

        # stop_expressions include branch conditions + general/terminal
        # stop conditions
        stop_expressions, stop_branch_order = self._get_stop_expressions(
            model, steptime0_sv
        )
        num_terminal_states = len(stop_expressions) - len(stop_branch_order)
        has_events = len(stop_expressions) > 0
        is_ode = len(model.algebraic) == 0

        state_labels = [to_variable_name(v.name) for v in states]
        initial_conditions = [model.initial_conditions[v] for v in states]
        if self._has_experiment:
            state_labels.append("steptime0")
            initial_conditions.append(pybamm.Scalar(0))
        diffeq = {}
        new_line = "\n"

        # inputs, find all pybamm.InputParameters in the model
        self._input_names = self._collect_input_names(all_vars, outputs)
        inputs = [vn for _, vn in self._input_names]

        if len(inputs) > 0:
            lines = ["in_i {"]
            for inpt in inputs:
                lines += [f"  {to_variable_name(inpt)} = 1,"]
            diffeq["in"] = new_line.join(lines) + new_line + "}"

        # extract constant vectors and matrices from model as tensors
        symbol_to_tensor_name = {}
        vectors = {}
        matrices = {}
        termination_events = stop_expressions
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

        if self._has_experiment and getattr(self, "_schedule_states", None):
            schedule_states = self._schedule_states
            sorted_by_index = sorted(schedule_states, key=lambda s: s.schedule_index)
            target_values = [
                f"{state.target_value:.{self.float_precision}g}"
                for state in sorted_by_index
            ]
            diffeq[DiffSLExport.STEP_VALUE_TENSOR_NAME] = self._tensor_block(
                DiffSLExport.STEP_VALUE_TENSOR_NAME, target_values
            )

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
        self._nstates = start_index

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
                    use_model_index=self._has_experiment,
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
                num_terminal_states=num_terminal_states,
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
            if self._has_experiment:
                eqn = f"{state_name_to_dstate_name('steptime0')}_i"
                lines += [f"  {eqn},"]
            diffeq["M"] = new_line.join(lines) + new_line + "}"

        # F
        model_index_gate = None
        if self._has_experiment:
            active_steps = len(stop_branch_order)
            gate_tensor_name = f"constant{tensor_index}"
            tensor_index += 1
            gate_entries = ["1.0"] * active_steps + ["0.0"] * num_terminal_states
            diffeq[gate_tensor_name] = self._tensor_block(
                gate_tensor_name, gate_entries
            )
            model_index_gate = f"{gate_tensor_name}_i[N]"
            lines = ["Fraw_i {"]
        else:
            lines = ["F_i {"]
        for rhs in model.rhs.values():
            eqn = equation_to_diffeq(
                rhs,
                y_slice_to_label,
                symbol_to_tensor_name,
                float_precision=self.float_precision,
                use_model_index=self._has_experiment,
            )
            lines += [f"  {eqn},"]
            start_index += rhs.size
        for algebraic in model.algebraic.values():
            eqn = equation_to_diffeq(
                algebraic,
                y_slice_to_label,
                symbol_to_tensor_name,
                float_precision=self.float_precision,
                use_model_index=self._has_experiment,
            )
            lines += [f"  {eqn},"]
            start_index += algebraic.size
        if self._has_experiment and model_index_gate is not None:
            # rhs for steptime0 is 1
            lines += ["  1.0,"]
            diffeq["F"] = (
                new_line.join(lines)
                + new_line
                + "}"
                + new_line
                + "F_i { "
                + model_index_gate
                + " * Fraw_i }"
                + new_line
            )
        else:
            diffeq["F"] = new_line.join(lines) + new_line + "}"

        # out
        lines = ["out_i {"]
        for output in outputs:
            eqn = equation_to_diffeq(
                all_vars[output],
                y_slice_to_label,
                symbol_to_tensor_name,
                float_precision=self.float_precision,
                use_model_index=self._has_experiment,
            )
            lines += [f"  {eqn},"]
        diffeq["out"] = new_line.join(lines) + new_line + "}"

        # events
        if has_events:
            if self._has_experiment:
                cycle_count = len(stop_branch_order)
                cycle_stops = stop_expressions[:cycle_count]
                general_stops = stop_expressions[cycle_count:]

                event_tensor_name = f"event{tensor_index}"
                tensor_index += 1
                event_entries = [
                    equation_to_diffeq(
                        expr,
                        y_slice_to_label,
                        symbol_to_tensor_name,
                        float_precision=self.float_precision,
                        use_model_index=self._has_experiment,
                    )
                    for expr in cycle_stops
                ]
                diffeq[event_tensor_name] = self._tensor_block(
                    event_tensor_name, event_entries
                )

                activestep_tensor_name = f"constant{tensor_index}"
                tensor_index += 1
                diffeq[activestep_tensor_name] = self._tensor_block(
                    activestep_tensor_name, stop_branch_order
                )

                # equality tensor of length equal to the number of steps
                # in a cycle. with a 1 at the index of the active step
                # and 0 elsewhere.
                equality_tensor_name = f"event{tensor_index}"
                tensor_index += 1
                a = f"{activestep_tensor_name}_i"
                equality_expr = f"heaviside({a} - N) - heaviside({a} - (N + 1))"
                diffeq[equality_tensor_name] = self._tensor_block(
                    equality_tensor_name, [equality_expr]
                )

                # gate the stop expressions so that only the active step's
                # stop expression is active
                gated_expr = f"({event_tensor_name}_i * {equality_tensor_name}_i + (1 - {equality_tensor_name}_i))"

                lines = ["stop_i {"]
                lines += [f"  {gated_expr},"]
                for stop_expr in general_stops:
                    eqn = equation_to_diffeq(
                        stop_expr,
                        y_slice_to_label,
                        symbol_to_tensor_name,
                        float_precision=self.float_precision,
                        use_model_index=self._has_experiment,
                    )
                    lines += [f"  {eqn},"]
                diffeq["stop"] = new_line.join(lines) + new_line + "}"
            else:
                lines = ["stop_i {"]
                for stop_expression in stop_expressions:
                    eqn = equation_to_diffeq(
                        stop_expression,
                        y_slice_to_label,
                        symbol_to_tensor_name,
                        float_precision=self.float_precision,
                        use_model_index=self._has_experiment,
                    )
                    lines += [f"  {eqn},"]
                diffeq["stop"] = new_line.join(lines) + new_line + "}"

        # reset - reset steptime0 to 0 at each step transition
        if self._has_experiment:
            n = total_model_state_size
            diffeq["reset"] = f"reset_i {{\n  u_i[0:{n}],\n  0.0,\n}}"

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
        if "reset" in diffeq:
            all_lines += [diffeq["reset"]]
        for key in diffeq.keys():
            if key.startswith("variable"):
                all_lines += [diffeq[key]]
        for key in out:
            all_lines += [diffeq[key]]

        return "\n".join(all_lines) + "\n"

    def default_inputs(self) -> dict:
        """
        Return a dict of default input values for the model.

        Returns
        -------
        dict
            PyBaMM-style parameter dict mapping parameter names to scalar values.
            The keys are the original PyBaMM parameter names (e.g. ``"Lower voltage cut-off [V]"``).
        """
        return {original_name: 1.0 for original_name, _ in self._input_names}

    def input_names(self) -> list[tuple[str, str]]:
        """
        Return a list of input parameter names for the model.

        Returns
        -------
        list[tuple[str, str]]
            List of tuples containing the original PyBaMM parameter name and its diffsl-transformed form (e.g. ``("Lower voltage cut-off [V]", "lowervoltagecutoffv")``).
        """
        return self._input_names

    def nstates(self) -> int | None:
        """
        Return the number of states in the model.

        Returns
        -------
        int
            The number of states in the model.
        """
        return self._nstates

    def map_inputs(self, inputs: dict) -> np.ndarray:
        """
        Map a PyBaMM inputs dict to the ordered array expected by the DiffSL model.

        The ordering matches the ``in_i {}`` block produced by :meth:`to_diffeq`.
        Each key in ``inputs`` may be either the original PyBaMM parameter name
        (e.g. ``"Lower voltage cut-off [V]"``) or its DiffSL-transformed form
        (e.g. ``"lowervoltagecutoffv"``).

        Parameters
        ----------
        inputs : dict
            PyBaMM-style parameter dict mapping parameter names to scalar values.
        outputs : list[str] or None, optional
            Output variable names, used to scan for ``InputParameter`` nodes that
            appear only inside output expressions.  Defaults to an empty list.

        Returns
        -------
        np.ndarray
            1-D float array ordered to match the ``in_i {}`` block of the exported
            DiffSL model.  Returns an empty array when the model has no input
            parameters.

        Raises
        ------
        KeyError
            If a required input parameter is not present in *inputs*.
        """
        ordered_names = self._input_names

        if not ordered_names:
            return np.array([], dtype=float)

        values = []
        for original_name, variable_name in ordered_names:
            if original_name in inputs:
                values.append(float(inputs[original_name]))
            elif variable_name in inputs:
                values.append(float(inputs[variable_name]))
            else:
                raise KeyError(
                    f"Input parameter '{original_name}' (DiffSL name: "
                    f"'{variable_name}') not found in inputs dict. "
                    f"Available keys: {list(inputs.keys())}"
                )
        return np.array(values, dtype=float)


def _equation_to_diffeq(
    equation: pybamm.Symbol,
    y_slice_to_label: dict[tuple[int], str],
    symbol_to_tensor_name: dict[pybamm.Symbol, str],
    float_precision: int = 20,
    transpose: bool = False,
    use_model_index: bool = False,
) -> str:
    if isinstance(equation, pybamm.Conditional) and equation in symbol_to_tensor_name:
        index = "j" if transpose else "i"
        return f"{symbol_to_tensor_name[equation]}_{index}[N]"
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
            use_model_index=use_model_index,
        )

        if equation.name == "@" and isinstance(equation.left, pybamm.Matrix):
            right = _equation_to_diffeq(
                equation.right,
                y_slice_to_label,
                symbol_to_tensor_name,
                float_precision=float_precision,
                transpose=not transpose,
                use_model_index=use_model_index,
            )
            return f"({left} * {right})"
        else:
            right = _equation_to_diffeq(
                equation.right,
                y_slice_to_label,
                symbol_to_tensor_name,
                float_precision=float_precision,
                transpose=transpose,
                use_model_index=use_model_index,
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
        return (
            f"{equation.name}("
            + _equation_to_diffeq(
                equation.child,
                y_slice_to_label,
                symbol_to_tensor_name,
                float_precision=float_precision,
                transpose=transpose,
                use_model_index=use_model_index,
            )
            + ")"
        )
    elif isinstance(equation, pybamm.Function):
        name = equation.function.__name__
        args = [
            _equation_to_diffeq(
                x,
                y_slice_to_label,
                symbol_to_tensor_name,
                float_precision=float_precision,
                transpose=transpose,
                use_model_index=use_model_index,
            )
            for x in equation.children
        ]
        if name == "_reg_power_evaluate":
            # Approximates |x|^a * sign(x) using:
            #  y = (x/scale) * ((x/scale)^2 + delta^2)^((a-1)/2) * scale^a
            delta = equation.delta
            x = args[0]
            a = args[1]
            # Pre-compute the inner exponent (a-1)/2 at code-gen time when possible,
            a_child = equation.children[1]
            if isinstance(a_child, pybamm.Scalar):
                a_val = float(a_child.evaluate())
                inner_exp = f"{(a_val - 1) / 2:.{float_precision}g}"
            else:
                inner_exp = f"({a} - 1) / 2"
            x = f"({x} / {args[2]})"
            return f"({x} * pow((pow({x}, 2) + {delta**2:.{float_precision}g}), {inner_exp})) * pow({args[2]}, {a})"
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
        name = to_variable_name(equation.name)
        if use_model_index and equation.name == pybamm.Simulation._STEP_VALUE_INPUT:
            return f"{name}_i[N]"
        return name
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
    use_model_index: bool = False,
) -> str:
    if not isinstance(equation, pybamm.Symbol):
        raise TypeError("equation must be a pybamm.Symbol")  # pragma: no cover
    return _equation_to_diffeq(
        equation,
        slice_to_label,
        symbol_to_tensor_name,
        float_precision=float_precision,
        transpose=transpose,
        use_model_index=use_model_index,
    )


def to_variable_name(name: str) -> str:
    """Convert a name to a valid diffeq variable name"""
    if name == pybamm.Simulation._STEP_VALUE_INPUT:
        return DiffSLExport.STEP_VALUE_TENSOR_NAME
    convert_to_underscore = [" ", "-", "(", ")", "[", "]", "{", "}", "/", "\\", "."]
    name = name.lower()
    for char in convert_to_underscore:
        name = name.replace(char, "")
    return name


def state_name_to_dstate_name(state_name: str) -> str:
    """Convert a state name to a dstate name"""
    return f"d{state_name}dt"
