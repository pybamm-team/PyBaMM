#
# Exception classes
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals


class DomainError(Exception):
    """Domain error: an operation was attempted on nodes with un-matched domains"""

    pass


class ModelError(Exception):
    """Model error: the model is not well-posed (can be before or after processing)"""

    pass
