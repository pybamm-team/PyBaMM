import lazy_loader


__getattr__, __dir__, __all__ = lazy_loader.attach(
    __name__,
    submodules={
        'print_name',
        'sympy_overrides',
    },
    submod_attrs={
        'print_name': [
            'GREEK_LETTERS',
            'PRINT_NAME_OVERRIDES',
            'prettify_print_name',
        ],
        'sympy_overrides': [
            'CustomPrint',
            'custom_print_func',
        ],
    },
)

__all__ = ['CustomPrint', 'GREEK_LETTERS', 'PRINT_NAME_OVERRIDES',
           'custom_print_func', 'prettify_print_name', 'print_name',
           'sympy_overrides']
