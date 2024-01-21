import lazy_loader


__getattr__, __dir__, __all__ = lazy_loader.attach(
    __name__,
    submodules={
        'pouch_cell_1D_current_collectors',
        'pouch_cell_2D_current_collectors',
        'x_full',
    },
    submod_attrs={
        'pouch_cell_1D_current_collectors': [
            'CurrentCollector1D',
        ],
        'pouch_cell_2D_current_collectors': [
            'CurrentCollector2D',
        ],
        'x_full': [
            'OneDimensionalX',
        ],
    },
)

__all__ = ['CurrentCollector1D', 'CurrentCollector2D', 'OneDimensionalX',
           'pouch_cell_1D_current_collectors',
           'pouch_cell_2D_current_collectors', 'x_full']
