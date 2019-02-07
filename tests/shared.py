#
# Shared methods and classes for testing
#
import pybamm

import numpy as np


class MeshForTesting(pybamm.BaseMesh):
    def __init__(self):
        super().__init__(None)
        self["whole cell"] = self.submeshclass(np.linspace(0, 1, 100))
        self["negative electrode"] = self.submeshclass(self["whole cell"].nodes[:30])
        self["separator"] = self.submeshclass(self["whole cell"].nodes[30:40])
        self["positive electrode"] = self.submeshclass(self["whole cell"].nodes[40:])
