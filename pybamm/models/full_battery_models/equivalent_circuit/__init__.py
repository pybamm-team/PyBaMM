import lazy_loader


__getattr__, __dir__, __all__ = lazy_loader.attach(
    __name__,
    submodules={
        'ecm_model_options',
        'thevenin',
    },
    submod_attrs={
        'ecm_model_options': [
            'NaturalNumberOption',
            'OperatingModes',
        ],
        'thevenin': [
            'Thevenin',
        ],
    },
)

__all__ = ['NaturalNumberOption', 'OperatingModes', 'Thevenin',
           'ecm_model_options', 'thevenin']
