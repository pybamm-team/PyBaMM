import lazy_loader


__getattr__, __dir__, __all__ = lazy_loader.attach(
    __name__,
    submodules={
        'base_electrolyte_diffusion',
        'constant_concentration',
        'full_diffusion',
        'leading_order_diffusion',
    },
    submod_attrs={
        'base_electrolyte_diffusion': [
            'BaseElectrolyteDiffusion',
        ],
        'constant_concentration': [
            'ConstantConcentration',
        ],
        'full_diffusion': [
            'Full',
        ],
        'leading_order_diffusion': [
            'LeadingOrder',
        ],
    },
)

__all__ = ['BaseElectrolyteDiffusion', 'ConstantConcentration', 'Full',
           'LeadingOrder', 'base_electrolyte_diffusion',
           'constant_concentration', 'full_diffusion',
           'leading_order_diffusion']
