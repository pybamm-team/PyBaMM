#
# Documentation utilities for actions on PyBaMM classes
#
from inspect import getmro


def copy_parameter_doc_from_parent(cls):
    """
    Add parameters from the base class to the docstring (Sphinx
    documentation). This constructs a new docstring by concatenating with
    formatting the base class docstring with the derived class. The
    base class docstring is truncated at the **Parameters** section.

    Usage: as a decorator @copy_parameter_doc_from_parent on derived
    class definition.
    """
    base_cls = getmro(cls)[1]
    parameters_section = "".join(base_cls.__doc__.partition("Parameters")[1:])
    cls.__doc__ += f"\n\n    {parameters_section}"
    return cls


# also implemented as a Sphinx extension in docs/sphinx_extend_parent.py, but this is
# not automatic, so we need to use the decorator manually in the class definition
def doc_extend_parent(cls):
    """
    Add a link to the base class in the docstring (Sphinx documentation) via the
    **Extends** directive. Constructs a new docstring element by concatenating
    with formatting the method resolution order (MRO) of the derived class.

    Usage: as a decorator @doc_extend_parent on derived class definition.
    """
    base_cls_name = f"{getmro(cls)[1].__module__}.{getmro(cls)[1].__name__}"
    cls.__doc__ += f"\n\n    **Extends:** :class:`{base_cls_name}`\n    "
    return cls
