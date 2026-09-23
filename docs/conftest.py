"""Apply the symbol mutation guard to documentation and notebook kernels."""

import os

os.environ["PYBAMM_TEST_FORBID_SYMBOL_MUTATION"] = "1"
