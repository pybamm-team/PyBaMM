#
# Tests for the LRUDict class
#
import unittest
from pybamm.solvers.lrudict import LRUDict
from collections import OrderedDict


class TestLRUDict(unittest.TestCase):
    def test_lrudict_defaultbehaviour(self):
        """Default behaviour [no LRU] mimics Dict"""
        d = LRUDict()
        dd = dict()
        for count in range(1, 100):
            d[count] = f"v{count}"
            dd[count] = f"v{count}"
            if count % 5 == 0:
                # LRU will reorder list, but not remove entries
                d.get(count - 2)
                dd.get(count - 2)
            # assertCountEqual checks that the same elements are present in
            # both lists, not just that the lists are of equal count
            self.assertCountEqual(set(d.keys()), set(dd.keys()))
            self.assertCountEqual(set(d.values()), set(dd.values()))

    def test_lrudict_noitems(self):
        """Edge case: no items in LRU, raises KeyError on assignment"""
        d = LRUDict(maxsize=-1)
        with self.assertRaises(KeyError):
            d["a"] = 1

    def test_lrudict_singleitem(self):
        """Only the last added element should ever be present"""
        d = LRUDict(maxsize=1)
        item_list = range(1, 100)
        self.assertEqual(len(d), 0)
        for item in item_list:
            d[item] = item
            self.assertEqual(len(d), 1)
            self.assertIsNotNone(d[item])
        # Finally, pop the only item and check that the dictionary is empty
        d.popitem()
        self.assertEqual(len(d), 0)

    def test_lrudict_multiitem(self):
        """Check that the correctly ordered items are always present"""
        for maxsize in range(1, 101, 5):
            d = LRUDict(maxsize=maxsize)
            expected = OrderedDict()
            for key in range(1, 100):
                value = f"v{key}"
                d[key] = value
                expected[key] = value
                if key % 5 == 0 and maxsize > 3:
                    # Push item to front of the list
                    key_to_read = key - 3
                    d.get(key_to_read)
                    expected.move_to_end(key_to_read)
                expected = OrderedDict(
                    (k, expected[k]) for k in list(expected.keys())[-maxsize:]
                )
                self.assertListEqual(list(d.keys()), list(expected.keys()))
                self.assertListEqual(list(d.values()), list(expected.values()))

    def test_lrudict_invalidkey(self):
        d = LRUDict()
        value = 1
        d["a"] = value
        # Access with valid key
        self.assertEqual(d["a"], value)  # checks getitem()
        self.assertEqual(d.get("a"), value)  # checks get()
        # Access with invalid key
        with self.assertRaises(KeyError):
            _ = d["b"]  # checks getitem()
        self.assertIsNone(d.get("b"))  # checks get()
