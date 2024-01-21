import lazy_loader


__getattr__, __dir__, __all__ = lazy_loader.attach(
    __name__,
    submodules={
        'base_active_material',
        'constant_active_material',
        'loss_active_material',
        'total_active_material',
    },
    submod_attrs={
        'base_active_material': [
            'BaseModel',
        ],
        'constant_active_material': [
            'Constant',
        ],
        'loss_active_material': [
            'LossActiveMaterial',
        ],
        'total_active_material': [
            'Total',
        ],
    },
)

__all__ = ['BaseModel', 'Constant', 'LossActiveMaterial', 'Total',
           'base_active_material', 'constant_active_material',
           'loss_active_material', 'total_active_material']
