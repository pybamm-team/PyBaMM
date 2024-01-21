import lazy_loader


__getattr__, __dir__, __all__ = lazy_loader.attach(
    __name__,
    submodules={
        'base_external_circuit',
        'explicit_control_external_circuit',
        'function_control_external_circuit',
    },
    submod_attrs={
        'base_external_circuit': [
            'BaseModel',
        ],
        'explicit_control_external_circuit': [
            'ExplicitCurrentControl',
            'ExplicitPowerControl',
            'ExplicitResistanceControl',
        ],
        'function_control_external_circuit': [
            'CCCVFunctionControl',
            'FunctionControl',
            'PowerFunctionControl',
            'ResistanceFunctionControl',
            'VoltageFunctionControl',
        ],
    },
)

__all__ = ['BaseModel', 'CCCVFunctionControl', 'ExplicitCurrentControl',
           'ExplicitPowerControl', 'ExplicitResistanceControl',
           'FunctionControl', 'PowerFunctionControl',
           'ResistanceFunctionControl', 'VoltageFunctionControl',
           'base_external_circuit', 'explicit_control_external_circuit',
           'function_control_external_circuit']
