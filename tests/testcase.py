#
# Custom TestCase class for pybamm
#
import unittest
import hashlib
from functools import wraps
from types import FunctionType
import numpy as np


def FixRandomSeed(method):
    """
    Wraps a method so that the random seed is set to a hash of the method name

    As the wrapper fixes the random seed before calling the method, tests can
    explicitly reinstate the random seed within their method bodies as desired,
    e.g. by calling np.random.seed(None) to restore normal behaviour.

    Generating a random seed from the method name allows particularly awkward
    sequences to be altered by changing the method name, such as by adding a
    trailing underscore, or other hash modifier, if required.
    """

    @wraps(method)
    def wrapped(*args, **kwargs):
        np.random.seed(
            int(hashlib.sha256(method.__name__.encode()).hexdigest(), 16) % (2**32)
        )
        return method(*args, **kwargs)

    return wrapped


class MakeAllTestsDeterministic(type):
    """
    Metaclass that wraps all class methods with FixRandomSeed()
    """

    def __new__(meta, classname, bases, classDict):
        newClassDict = {}
        for attributeName, attribute in classDict.items():
            if isinstance(attribute, FunctionType):
                attribute = FixRandomSeed(attribute)
            newClassDict[attributeName] = attribute
        return type.__new__(meta, classname, bases, newClassDict)


class TestCase(unittest.TestCase, metaclass=MakeAllTestsDeterministic):
    """
    Custom TestCase class for pybamm
    """

    def assertDomainEqual(self, a, b):
        "Check that two domains are equal, ignoring empty domains"
        a_dict = {k: v for k, v in a.items() if v != []}
        b_dict = {k: v for k, v in b.items() if v != []}
        self.assertEqual(a_dict, b_dict)
