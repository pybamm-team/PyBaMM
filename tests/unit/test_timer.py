#
# Tests the Timer class.
#
# The code in this file is adapted from Pints
# (see https://github.com/pints-team/pints)
#
import pybamm


class TestTimer:
    """
    Tests the basic methods of the Timer class.
    """

    def __init__(self, name):
        super().__init__(name)

    def test_timing(self):
        t = pybamm.Timer()
        a = t.time().value
        assert a >= 0
        for _ in range(100):
            assert t.time().value > a
        a = t.time().value
        t.reset()
        b = t.time().value
        assert b >= 0
        assert b < a

    def test_timer_format(self):
        assert str(pybamm.TimerTime(1e-9)) == "1.000 ns"
        assert str(pybamm.TimerTime(0.000000123456789)) == "123.457 ns"
        assert str(pybamm.TimerTime(1e-6)) == "1.000 us"
        assert str(pybamm.TimerTime(0.000123456789)) == "123.457 us"
        assert str(pybamm.TimerTime(0.999e-3)) == "999.000 us"
        assert str(pybamm.TimerTime(1e-3)) == "1.000 ms"
        assert str(pybamm.TimerTime(0.123456789)) == "123.457 ms"
        assert str(pybamm.TimerTime(2)) == "2.000 s"
        assert str(pybamm.TimerTime(2.5)) == "2.500 s"
        assert str(pybamm.TimerTime(12.5)) == "12.500 s"
        assert str(pybamm.TimerTime(59.41)) == "59.410 s"
        assert str(pybamm.TimerTime(59.4126347547)) == "59.413 s"
        assert str(pybamm.TimerTime(60.2)) == "1 minute, 0 seconds"
        assert str(pybamm.TimerTime(61)) == "1 minute, 1 second"
        assert str(pybamm.TimerTime(121)) == "2 minutes, 1 second"
        assert (
            str(pybamm.TimerTime(604800))
            == "1 week, 0 days, 0 hours, 0 minutes, 0 seconds"
        )
        assert (
            str(pybamm.TimerTime(2 * 604800 + 3 * 3600 + 60 + 4))
            == "2 weeks, 0 days, 3 hours, 1 minute, 4 seconds"
        )

        assert repr(pybamm.TimerTime(1.5)) == "pybamm.TimerTime(1.5)"

    def test_timer_operations(self):
        assert (pybamm.TimerTime(1) + 2).value == 3
        assert (1 + pybamm.TimerTime(1)).value == 2
        assert (pybamm.TimerTime(1) - 2).value == -1
        assert (pybamm.TimerTime(1) - pybamm.TimerTime(2)).value == -1
        assert (1 - pybamm.TimerTime(1)).value == 0
        assert (pybamm.TimerTime(4) * 2).value == 8
        assert (pybamm.TimerTime(4) * pybamm.TimerTime(2)).value == 8
        assert (2 * pybamm.TimerTime(5)).value == 10
        assert (pybamm.TimerTime(4) / 2).value == 2
        assert (pybamm.TimerTime(4) / pybamm.TimerTime(2)).value == 2
        assert (2 / pybamm.TimerTime(5)).value == 2 / 5

        assert pybamm.TimerTime(1) == pybamm.TimerTime(1)
        assert pybamm.TimerTime(1) != pybamm.TimerTime(2)
