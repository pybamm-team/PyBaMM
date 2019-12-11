#
# Simulate a GITT experiment
#
from .experiment import Experiment


class GITT(Experiment):
    def __init__(
        self, discharge_time=1, discharge_current=1, rest_time=1, num_cycles=None
    ):
        self.discharge_time = discharge_time
