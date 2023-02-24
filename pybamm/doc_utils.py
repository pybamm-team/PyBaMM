#
# Documentation utilities for actions on PyBaMM classes
#
from inspect import getmro


def parameters_from(cls):
    """
    Add parameters from the base class to the docstring (Sphinx
    documentation). This constructs a new docstring by concatenating with
    formatting the base class docstring with the derived class. The
    base class docstring is truncated at the **Parameters** section.

    Usage: as a decorator @parameters_from on derived class definition.
    """
    base_cls = getmro(cls)[1]
    cls.__doc__ += "\n\n    " + "".join(base_cls.__doc__.partition("Parameters")[1:])
    return cls


def extends(cls):
    """
    Add a link to the base class in the docstring (Sphinx documentation) via the
    **Extends** directive. Constructs a new docstring element by concatenating
    with formatting the method resolution order (MRO) of the derived class.

    Usage: as a decorator @extends on derived class definition.
    """
    base_cls_name = getmro(cls)[1].__module__ + "." + getmro(cls)[1].__name__
    cls.__doc__ += "\n\n    " + "**Extends:** :class:`" + base_cls_name + "`\n    "
    return cls
