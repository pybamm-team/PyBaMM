import lazy_loader


__getattr__, __dir__, __all__ = lazy_loader.attach(
    __name__,
    submodules={
        'experiment',
        'step',
    },
    submod_attrs={
        'experiment': [
            'Experiment',
        ],
        'step': [
            'BaseTermination',
            'CrateTermination',
            'CurrentTermination',
            'CustomTermination',
            'VoltageTermination',
            'c_rate',
            'current',
            'power',
            'resistance',
            'rest',
            'step_termination',
            'steps',
            'string',
            'voltage',
        ],
    },
)

__all__ = ['BaseTermination', 'CrateTermination', 'CurrentTermination',
           'CustomTermination', 'Experiment', 'VoltageTermination', 'c_rate',
           'current', 'experiment', 'power', 'resistance', 'rest', 'step',
           'step_termination', 'steps', 'string', 'voltage']
