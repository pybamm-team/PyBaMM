#
# Utility classes for PyBaMM
#
# The code in this file is adapted from Pints
# (see https://github.com/pints-team/pints)
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals

import timeit
import cProfile
import pstats
import os


class Timer(object):
    """
    Provides accurate timing.

    Example
    -------
    timer = pybamm.Timer()
    print(timer.format(timer.time()))

    """

    def __init__(self):
        self._start = timeit.default_timer()

    def format(self, time=None):
        """
        Formats a (non-integer) number of seconds, returns a string like
        "5 weeks, 3 days, 1 hour, 4 minutes, 9 seconds", or "0.0019 seconds".

        Arguments
        ---------
        time : float, optional
            The time to be formatted.

        Returns
        -------
        string
            The string representation of ``time`` in human-readable form.
        """
        if time is None:
            time = self.time()
        if time < 1e-2:
            return str(time) + " seconds"
        elif time < 60:
            return str(round(time, 2)) + " seconds"
        output = []
        time = int(round(time))
        units = [(604800, "week"), (86400, "day"), (3600, "hour"), (60, "minute")]
        for k, name in units:
            f = time // k
            if f > 0 or output:
                output.append(str(f) + " " + (name if f == 1 else name + "s"))
            time -= f * k
        output.append("1 second" if time == 1 else str(time) + " seconds")
        return ", ".join(output)

    def reset(self):
        """
        Resets this timer's start time.
        """
        self._start = timeit.default_timer()

    def time(self):
        """
        Returns the time (float, in seconds) since this timer was created,
        or since meth:`reset()` was last called.
        """
        return timeit.default_timer() - self._start


def profile(code, sort="cumulative", num=30):
    """Common-use for cProfile"""
    cProfile.run(code)
    stats = pstats.Stats()
    stats.sort_stats(sort)
    return stats


def load_function(filename):
    """
    Load a python function from a file "function_name.py" called "function_name".

    Arguments
    ---------
    filename : str
        The name of the file containing the function of the same name.

    Returns
    -------
    function
        The python function loaded from the file.
    """

    if not filename.endswith('.py'):
        raise ValueError("Expected filename.py, but got {}".format(filename))

    func_to_return = abs

    return func_to_return
