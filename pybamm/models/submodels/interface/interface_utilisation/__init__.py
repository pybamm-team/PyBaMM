import lazy_loader


__getattr__, __dir__, __all__ = lazy_loader.attach(
    __name__,
    submodules={
        'base_utilisation',
        'constant_utilisation',
        'current_driven_utilisation',
        'full_utilisation',
    },
    submod_attrs={
        'base_utilisation': [
            'BaseModel',
        ],
        'constant_utilisation': [
            'Constant',
        ],
        'current_driven_utilisation': [
            'CurrentDriven',
        ],
        'full_utilisation': [
            'Full',
        ],
    },
)

__all__ = ['BaseModel', 'Constant', 'CurrentDriven', 'Full',
           'base_utilisation', 'constant_utilisation',
           'current_driven_utilisation', 'full_utilisation']
