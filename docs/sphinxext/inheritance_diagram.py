# Sphinx extension to add an inheritance diagram in the docstring of a class built upon
# the built-in sphinxext.inheritance_diagram extension. The inheritance diagram is
# generated via graphviz and the fully qualified name of the class.

from inspect import getmro, isclass


def add_diagram(app, what, name, obj, options, lines):
    # if it is not a class, do nothing
    if not isclass(obj):
        return
    # if it is a model or submodel class, add the inheritance diagram
    else:
        # get the fully qualified name of the class
        cls_name = f"{obj.__module__}.{obj.__qualname__}"
        # check if the class is a model or submodel
        if "pybamm.models" in cls_name:
            # check if the class derives from another class
            if not len(getmro(obj)) > 2:
                # do nothing if it is not a derived class
                return

            # Append the inheritance diagram to the docstring
            lines.append("\n")
            lines.append(".. dropdown:: View inheritance diagram for this model")
            lines.append("   :animate: fade-in-slide-down")
            lines.append("   :icon: eye\n")
            lines.append("   :class-title: sd-align-major-center sd-fs-6 \n")
            lines.append("   :class-container: sd-text-info \n")
            lines.append("\n")
            lines.append("       .. inheritance-diagram:: " + cls_name)
            lines.append("           :parts: 2\n")
            lines.append("\n")


def setup(app):
    app.connect("autodoc-process-docstring", add_diagram)
    return {
        "version": "0.1.0",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
