#
# Unit tests for doc_utils. These do not test how the actual documentation
# is generated, but rather that the docstrings are correctly modified
#

import pybamm
import unittest
from inspect import getmro
from pybamm.doc_utils import copy_parameter_doc_from_parent, doc_extend_parent


class TestDocUtils(unittest.TestCase):
    def test_copy_parameter_doc_from_parent(self):
        """Test if parameters from the parent class are copied to
        child class docstring"""

        class Base:
            """Base class for testing docstring parameter inheritance

            Parameters:
            ----------
            foo : str
                description for foo
            bar : str
                description for bar."""

            def __init__(self, foo, bar):
                pass

        @copy_parameter_doc_from_parent
        class Derived(Base):
            """Derived class for testing docstring parameter inheritance"""

            def __init__(self, foo, bar):
                super().__init__(foo, bar)

        base_parameters = "".join(Base.__doc__.partition("Parameters")[1:])
        derived_parameters = "".join(Derived.__doc__.partition("Parameters")[1:])
        # check that the parameters section is in the docstring
        self.assertMultiLineEqual(base_parameters, derived_parameters)

    def test_doc_extend_parent(self):
        """Test if the child class has the Extends directive in its docstring"""

        class Base:
            """Base class for testing doc_utils.doc_extend_parent"""

            def __init__(self, foo, bar):
                pass

        @doc_extend_parent
        class Derived(Base):
            """Derived class for testing doc_utils.doc_extend_parent"""

            def __init__(self, param):
                super().__init__(param)

        # check that the Extends directive is in the docstring
        self.assertIn("**Extends:**", Derived.__doc__)

        # check that the Extends directive maps to the correct base class
        base_cls_name = f"{getmro(Derived)[1].__module__}.{getmro(Derived)[1].__name__}"
        self.assertEqual(
            Derived.__doc__.partition("**Extends:**")[2].strip(),
            f":class:`{base_cls_name}`",
        )


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
