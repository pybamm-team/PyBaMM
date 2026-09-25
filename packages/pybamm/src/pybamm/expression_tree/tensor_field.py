"""
TensorField class for representing tensor fields in PyBaMM.
"""

from __future__ import annotations

import pybamm


class TensorField(pybamm.Symbol):
    """
    A node in the expression tree representing a tensor field.

    TensorField can represent tensors of rank 1 (vectors) or rank 2 (matrices).
    Components are stored in a nested list structure.

    Parameters
    ----------
    components : list
        For rank-1: list of symbols [c0, c1]
        For rank-2: list of lists [[c00, c01], [c10, c11]]
    domain : list, optional
        Domain of the tensor field. If not provided, inferred from components.
    """

    __slots__ = ("_rank", "_shape")

    def __init__(self, components, domain=None, name="tensor_field"):
        # Determine rank and shape from components structure
        if not components:
            raise ValueError("Components cannot be empty")

        if isinstance(components[0], list):
            # Rank-2 tensor
            self._rank = 2
            self._shape = (len(components), len(components[0]))
            # Validate all rows have same length
            for i, row in enumerate(components):
                if len(row) != self._shape[1]:
                    raise ValueError(
                        f"Row {i} has {len(row)} elements, expected {self._shape[1]}"
                    )
            # Flatten for children
            children = [c for row in components for c in row]
        else:
            # Rank-1 tensor
            self._rank = 1
            self._shape = (len(components),)
            children = list(components)

        # Infer domain from first component if not provided
        if domain is None:
            first_component = children[0] if children else None
            if first_component is not None and hasattr(first_component, "domain"):
                domain = first_component._domains["primary"]

        # Validate all components have same domain
        for child in children:
            if hasattr(child, "domain") and child._domains["primary"] != domain:
                raise ValueError(
                    f"All components must have the same domain. "
                    f"Expected {domain}, got {child._domains['primary']}"
                )

        super().__init__(name=name, children=children, domain=domain)

    @property
    def rank(self):
        """Return the rank of the tensor (1 for vector, 2 for matrix)."""
        return self._rank

    @property
    def shape(self):
        """Return the shape of the tensor as a tuple."""
        return self._shape

    @property
    def components(self):
        """Return the components in their nested structure."""
        children = self._children
        if self._rank == 1:
            return list(children)
        n_columns = self._shape[1]
        return [
            list(children[row * n_columns : (row + 1) * n_columns])
            for row in range(self._shape[0])
        ]

    def __getitem__(self, idx):
        """Access components by index.

        For rank-1: single index (e.g., tensor[0])
        For rank-2: tuple index (e.g., tensor[0, 1]) or single index for row
        """
        if self._rank == 1:
            if isinstance(idx, tuple):
                if len(idx) != 1:
                    raise IndexError(
                        f"Too many indices for rank-1 tensor: got {len(idx)}, expected 1"
                    )
                idx = idx[0]
            return self.components[idx]
        else:  # rank == 2
            if isinstance(idx, tuple):
                if len(idx) == 1:
                    return self.components[idx[0]]
                elif len(idx) == 2:
                    return self.components[idx[0]][idx[1]]
                else:
                    raise IndexError(
                        f"Too many indices for rank-2 tensor: got {len(idx)}, expected <= 2"
                    )
            else:
                # Single index returns a row
                return self.components[idx]

    def create_copy(
        self,
        new_children: list[pybamm.Symbol] | None = None,
        perform_simplifications: bool = True,
    ):
        """Create a copy of this tensor field with optional new children."""
        if new_children is None:
            # Copy all children
            new_children = [
                child.create_copy(perform_simplifications=perform_simplifications)
                for child in self._children
            ]

        # Reconstruct the nested structure
        if self._rank == 1:
            new_components = new_children
        else:
            # Reshape flat list back to nested structure
            new_components = []
            idx = 0
            for _i in range(self._shape[0]):
                row = []
                for _j in range(self._shape[1]):
                    row.append(new_children[idx])
                    idx += 1
                new_components.append(row)

        return TensorField(new_components, domain=self._domains["primary"])

    def to_json(self):
        return {
            "name": self.name,
            "domains": self._domains,
            "children": list(self.components),
        }

    @classmethod
    def _from_json(cls, snippet):
        return cls(snippet["children"])

    def _evaluate_for_shape(self):
        """Delegate shape evaluation to first component."""
        return self._children[0].evaluate_for_shape()

    def evaluates_on_edges(self, dimension: str) -> bool:
        """Check if any component evaluates on edges.

        Returns True if all components evaluate on edges,
        False if none do, raises error if mixed.
        """
        edge_status = [child.evaluates_on_edges(dimension) for child in self._children]

        if all(edge_status):
            return True
        elif not any(edge_status):
            return False
        else:
            raise ValueError(
                "All tensor components must either all evaluate on edges or none"
            )
