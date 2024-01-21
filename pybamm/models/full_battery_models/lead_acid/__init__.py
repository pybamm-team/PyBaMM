#
# Root of the lead-acid models module.
#
import lazy_loader


__getattr__, __dir__, __all__ = lazy_loader.attach(
    __name__,
    submodules={
        'base_lead_acid_model',
        'basic_full',
        'full',
        'loqs',
    },
    submod_attrs={
        'base_lead_acid_model': [
            'BaseModel',
        ],
        'basic_full': [
            'BasicFull',
        ],
        'full': [
            'Full',
        ],
        'loqs': [
            'LOQS',
        ],
    },
)

__all__ = ['BaseModel', 'BasicFull', 'Full', 'LOQS', 'base_lead_acid_model',
           'basic_full', 'full', 'loqs']
