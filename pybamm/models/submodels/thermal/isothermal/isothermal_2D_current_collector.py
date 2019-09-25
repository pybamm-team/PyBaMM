#
# Class for isothermal case which accounts for current collectors
#
from .base_isothermal import BaseModel


class CurrentCollector2D(BaseModel):
    """Class for isothermal submodel with a 2D current collector"""

    def __init__(self, param):
        super().__init__(param)
