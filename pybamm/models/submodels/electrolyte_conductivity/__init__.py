import lazy_loader


__getattr__, __dir__, __all__ = lazy_loader.attach(
    __name__,
    submodules={
        'base_electrolyte_conductivity',
        'composite_conductivity',
        'full_conductivity',
        'integrated_conductivity',
        'leading_order_conductivity',
        'surface_potential_form',
    },
    submod_attrs={
        'base_electrolyte_conductivity': [
            'BaseElectrolyteConductivity',
        ],
        'composite_conductivity': [
            'Composite',
        ],
        'full_conductivity': [
            'Full',
        ],
        'integrated_conductivity': [
            'Integrated',
        ],
        'leading_order_conductivity': [
            'LeadingOrder',
        ],
        'surface_potential_form': [
            'BaseLeadingOrderSurfaceForm',
            'BaseModel',
            'CompositeAlgebraic',
            'CompositeDifferential',
            'Explicit',
            'FullAlgebraic',
            'FullDifferential',
            'LeadingOrderAlgebraic',
            'LeadingOrderDifferential',
            'composite_surface_form_conductivity',
            'explicit_surface_form_conductivity',
            'full_surface_form_conductivity',
            'leading_surface_form_conductivity',
        ],
    },
)

__all__ = ['BaseElectrolyteConductivity', 'BaseLeadingOrderSurfaceForm',
           'BaseModel', 'Composite', 'CompositeAlgebraic',
           'CompositeDifferential', 'Explicit', 'Full', 'FullAlgebraic',
           'FullDifferential', 'Integrated', 'LeadingOrder',
           'LeadingOrderAlgebraic', 'LeadingOrderDifferential',
           'base_electrolyte_conductivity', 'composite_conductivity',
           'composite_surface_form_conductivity',
           'explicit_surface_form_conductivity', 'full_conductivity',
           'full_surface_form_conductivity', 'integrated_conductivity',
           'leading_order_conductivity', 'leading_surface_form_conductivity',
           'surface_potential_form']
