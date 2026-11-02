import importlib.util as il
from pathlib import Path

from .version import __version__

idaklu_spec = il.find_spec("pybammsolvers.idaklu")
idaklu = il.module_from_spec(idaklu_spec)
idaklu_spec.loader.exec_module(idaklu)


def get_include() -> str:
    """Directory of the C headers for building block solver plugins."""
    return str(Path(__file__).parent / "include")
