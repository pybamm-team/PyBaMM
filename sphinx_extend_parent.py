# Sphinx extension to add a link to a base class in the docstring of a derived class
# via the **Extends** directive. The base class is determined by the method resolution
# order (MRO) of the derived class.

from inspect import getmro, isclass


def process_docstring(app, what, name, obj, options, lines):
    # if it is not a class, do nothing
    if not isclass(obj):
        return
    # else if it is a class
    else:
        # check if the class derives from another class
        if not len(getmro(obj)) > 2:
            # do nothing if it is not a derived class
            return
        # check if the class has a docstring
        elif lines:
            # get the base class name
            base_cls_name = f"{getmro(obj)[1].__module__}.{getmro(obj)[1].__name__}"
            # add the extends keyword to the docstring
            lines.append(f"**Extends:** :class:`{base_cls_name}`")


def setup(app):
    app.connect("autodoc-process-docstring", process_docstring)
