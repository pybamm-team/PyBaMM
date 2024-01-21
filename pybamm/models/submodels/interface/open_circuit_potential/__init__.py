import lazy_loader


__getattr__, __dir__, __all__ = lazy_loader.attach(
    __name__,
    submodules={
        'base_ocp',
        'current_sigmoid_ocp',
        'msmr_ocp',
        'single_ocp',
    },
    submod_attrs={
        'base_ocp': [
            'BaseOpenCircuitPotential',
        ],
        'current_sigmoid_ocp': [
            'CurrentSigmoidOpenCircuitPotential',
        ],
        'msmr_ocp': [
            'MSMROpenCircuitPotential',
        ],
        'single_ocp': [
            'SingleOpenCircuitPotential',
        ],
    },
)

__all__ = ['BaseOpenCircuitPotential', 'CurrentSigmoidOpenCircuitPotential',
           'MSMROpenCircuitPotential', 'SingleOpenCircuitPotential',
           'base_ocp', 'current_sigmoid_ocp', 'msmr_ocp', 'single_ocp']
