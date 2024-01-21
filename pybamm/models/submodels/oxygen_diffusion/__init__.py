import lazy_loader


__getattr__, __dir__, __all__ = lazy_loader.attach(
    __name__,
    submodules={
        'base_oxygen_diffusion',
        'full_oxygen_diffusion',
        'leading_oxygen_diffusion',
        'no_oxygen',
    },
    submod_attrs={
        'base_oxygen_diffusion': [
            'BaseModel',
        ],
        'full_oxygen_diffusion': [
            'Full',
        ],
        'leading_oxygen_diffusion': [
            'LeadingOrder',
        ],
        'no_oxygen': [
            'NoOxygen',
        ],
    },
)

__all__ = ['BaseModel', 'Full', 'LeadingOrder', 'NoOxygen',
           'base_oxygen_diffusion', 'full_oxygen_diffusion',
           'leading_oxygen_diffusion', 'no_oxygen']
