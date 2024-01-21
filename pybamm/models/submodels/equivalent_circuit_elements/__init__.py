import lazy_loader


__getattr__, __dir__, __all__ = lazy_loader.attach(
    __name__,
    submodules={
        'ocv_element',
        'rc_element',
        'resistor_element',
        'thermal',
        'voltage_model',
    },
    submod_attrs={
        'ocv_element': [
            'OCVElement',
        ],
        'rc_element': [
            'RCElement',
        ],
        'resistor_element': [
            'ResistorElement',
        ],
        'thermal': [
            'ThermalSubModel',
        ],
        'voltage_model': [
            'VoltageModel',
        ],
    },
)

__all__ = ['OCVElement', 'RCElement', 'ResistorElement', 'ThermalSubModel',
           'VoltageModel', 'ocv_element', 'rc_element', 'resistor_element',
           'thermal', 'voltage_model']
