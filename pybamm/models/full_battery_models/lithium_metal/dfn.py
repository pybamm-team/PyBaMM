#
# Doyle-Fuller-Newman (DFN) Model
#
import pybamm
from ..lithium_ion.dfn import DFN as LithiumIonDFN


class DFN(LithiumIonDFN):
    """
    Doyle-Fuller-Newman (DFN) model of a lithium-metal battery. This model reuses the
    submodels from the lithium-ion DFN, and adds some submodels specific to
    lithium-metal.

    Parameters
    ----------
    options : dict, optional
        A dictionary of options to be passed to the model. For a detailed list of
        options see :class:`~pybamm.BatteryModelOptions`.
    name : str, optional
        The name of the model.
    build :  bool, optional
        Whether to build the model on instantiation. Default is True. Setting this
        option to False allows users to change any number of the submodels before
        building the complete model (submodels cannot be changed after the model is
        built).
    """

    def __init__(
        self, options=None, name="Doyle-Fuller-Newman lithium metal model", build=True
    ):
        options = options or {}
        options["working electrode"] = "positive"
        super().__init__(options, name, build=False)

        if build:
            self.build_model()

    def set_lithium_metal_electrode_submodel(self):
        self.submodels["lithium metal electrode"] = pybamm.lithium_metal_electrode.Full(
            self.param, self.options
        )
