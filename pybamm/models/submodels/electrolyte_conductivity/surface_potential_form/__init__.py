# Full order models
from .full_surface_form_conductivity import FullAlgebraic, FullDifferential

# Composite models
from .composite_surface_form_conductivity import (
    CompositeDifferential,
    CompositeAlgebraic,
)

# Leading-order models
from .leading_surface_form_conductivity import (
    LeadingOrderDifferential,
    LeadingOrderAlgebraic,
)

# Explicit model
import lazy_loader


__getattr__, __dir__, __all__ = lazy_loader.attach(
    __name__,
    submodules={
        'composite_surface_form_conductivity',
        'explicit_surface_form_conductivity',
        'full_surface_form_conductivity',
        'leading_surface_form_conductivity',
    },
    submod_attrs={
        'composite_surface_form_conductivity': [
            'BaseModel',
            'CompositeAlgebraic',
            'CompositeDifferential',
        ],
        'explicit_surface_form_conductivity': [
            'Explicit',
        ],
        'full_surface_form_conductivity': [
            'BaseModel',
            'FullAlgebraic',
            'FullDifferential',
        ],
        'leading_surface_form_conductivity': [
            'BaseLeadingOrderSurfaceForm',
            'LeadingOrderAlgebraic',
            'LeadingOrderDifferential',
        ],
    },
)

__all__ = ['BaseLeadingOrderSurfaceForm', 'BaseModel', 'CompositeAlgebraic',
           'CompositeDifferential', 'Explicit', 'FullAlgebraic',
           'FullDifferential', 'LeadingOrderAlgebraic',
           'LeadingOrderDifferential', 'composite_surface_form_conductivity',
           'explicit_surface_form_conductivity',
           'full_surface_form_conductivity',
           'leading_surface_form_conductivity']
