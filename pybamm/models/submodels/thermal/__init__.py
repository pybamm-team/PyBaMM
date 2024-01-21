import lazy_loader


__getattr__, __dir__, __all__ = lazy_loader.attach(
    __name__,
    submodules={
        'base_thermal',
        'isothermal',
        'lumped',
        'pouch_cell',
    },
    submod_attrs={
        'base_thermal': [
            'BaseThermal',
        ],
        'isothermal': [
            'Isothermal',
        ],
        'lumped': [
            'Lumped',
        ],
        'pouch_cell': [
            'CurrentCollector1D',
            'CurrentCollector2D',
            'OneDimensionalX',
            'pouch_cell_1D_current_collectors',
            'pouch_cell_2D_current_collectors',
            'x_full',
        ],
    },
)

__all__ = ['BaseThermal', 'CurrentCollector1D', 'CurrentCollector2D',
           'Isothermal', 'Lumped', 'OneDimensionalX', 'base_thermal',
           'isothermal', 'lumped', 'pouch_cell',
           'pouch_cell_1D_current_collectors',
           'pouch_cell_2D_current_collectors', 'x_full']
