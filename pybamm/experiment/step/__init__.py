import lazy_loader


__getattr__, __dir__, __all__ = lazy_loader.attach(
    __name__,
    submodules={
        'step_termination',
        'steps',
    },
    submod_attrs={
        'step_termination': [
            'BaseTermination',
            'CrateTermination',
            'CurrentTermination',
            'CustomTermination',
            'VoltageTermination',
        ],
        'steps': [
            'c_rate',
            'current',
            'power',
            'resistance',
            'rest',
            'string',
            'voltage',
        ],
    },
)

__all__ = ['BaseTermination', 'CrateTermination', 'CurrentTermination',
           'CustomTermination', 'VoltageTermination', 'c_rate', 'current',
           'power', 'resistance', 'rest', 'step_termination', 'steps',
           'string', 'voltage']
