import importlib.util as il


idaklu_spec = il.find_spec("pybammsolvers.idaklu")
idaklu = il.module_from_spec(idaklu_spec)
idaklu_spec.loader.exec_module(idaklu)
