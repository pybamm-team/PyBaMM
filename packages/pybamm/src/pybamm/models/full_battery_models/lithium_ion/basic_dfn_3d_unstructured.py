#
# Basic Doyle-Fuller-Newman (DFN) Model — 3D Unstructured FVM
#
from __future__ import annotations

from pybamm.models.full_battery_models.lithium_ion.basic_dfn_2d_unstructured import (
    BasicDFN2DUnstructured,
)


class BasicDFN3DUnstructured(BasicDFN2DUnstructured):
    """Doyle-Fuller-Newman (DFN) model on a 3D unstructured mesh.

    Extends :class:`BasicDFN2DUnstructured` to three spatial dimensions
    (x, y, z) on hexahedral or tetrahedral elements.  The through-cell
    direction is *x*, the width direction is *y*, and the height direction
    is *z*.

    Parameters
    ----------
    name : str, optional
        The name of the model.
    element_type : str, optional
        Element type for the built-in mesh generator: ``"hexahedron"``
        (default, TPFA-orthogonal) or ``"tetrahedron"``.
    """

    _three_dimensional = True
    _default_var_pts = {
        "x_n": 10,
        "x_s": 10,
        "x_p": 10,
        "r_p": 20,
        "r_n": 20,
        "y": 5,
        "z": 5,
    }

    def __init__(
        self,
        name: str = "Doyle-Fuller-Newman model (3D unstructured)",
        element_type: str = "hexahedron",
    ):
        super().__init__(name=name, element_type=element_type)
