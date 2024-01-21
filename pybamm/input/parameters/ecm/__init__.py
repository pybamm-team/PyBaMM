import lazy_loader


__getattr__, __dir__, __all__ = lazy_loader.attach(
    __name__,
    submodules={
        'example_set',
    },
    submod_attrs={
        'example_set': [
            'c1',
            'c1_data',
            'dUdT',
            'dUdT_data',
            'get_parameter_values',
            'ocv',
            'ocv_data',
            'r0',
            'r0_data',
            'r1',
            'r1_data',
        ],
    },
)

__all__ = ['c1', 'c1_data', 'dUdT', 'dUdT_data', 'example_set',
           'get_parameter_values', 'ocv', 'ocv_data', 'r0', 'r0_data', 'r1',
           'r1_data']
