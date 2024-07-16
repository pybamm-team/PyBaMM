#
# Test base submodel
#
import pytest
import pybamm


class TestBaseSubModel:
    def test_domain(self):
        # Accepted string
        submodel = pybamm.BaseSubModel(None, "negative", phase="primary")
        assert submodel.domain == "negative"

        # None
        submodel = pybamm.BaseSubModel(None, None)
        assert submodel.domain is None

        # bad string
        with pytest.raises(pybamm.DomainError):
            pybamm.BaseSubModel(None, "bad string")

    def test_phase(self):
        # Without domain
        submodel = pybamm.BaseSubModel(None, None)
        assert submodel.phase is None

        with pytest.raises(ValueError, match="Phase must be None"):
            pybamm.BaseSubModel(None, None, phase="primary")

        # With domain
        submodel = pybamm.BaseSubModel(None, "negative", phase="primary")
        assert submodel.phase == "primary"
        assert submodel.phase_name == ""

        submodel = pybamm.BaseSubModel(
            None, "negative", options={"particle phases": "2"}, phase="secondary"
        )
        assert submodel.phase == "secondary"
        assert submodel.phase_name == "secondary "

        with pytest.raises(ValueError, match="Phase must be 'primary'"):
            pybamm.BaseSubModel(None, "negative", phase="secondary")
        with pytest.raises(ValueError, match="Phase must be either 'primary'"):
            pybamm.BaseSubModel(
                None, "negative", options={"particle phases": "2"}, phase="tertiary"
            )
        with pytest.raises(ValueError, match="Phase must be 'primary'"):
            # 2 phases in the negative but only 1 in the positive
            pybamm.BaseSubModel(
                None,
                "positive",
                options={"particle phases": ("2", "1")},
                phase="secondary",
            )
