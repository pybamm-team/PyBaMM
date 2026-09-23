from collections import OrderedDict


class LRUDict(OrderedDict):
    """LRU extension of a dictionary"""

    def __init__(self, maxsize=None):
        """maxsize limits the item count based on an LRU strategy

        The dictionary remains unbound when maxsize = 0 | None
        """
        super().__init__()
        self.maxsize = maxsize

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        while self.maxsize and self.__len__() > self.maxsize:
            self.popitem(last=False)

    def __getitem__(self, key):
        try:
            self.move_to_end(key, last=True)
        except KeyError:
            pass  # Allow parent to handle exception
        return super().__getitem__(key)

    def get(self, key):
        try:
            self.move_to_end(key, last=True)
        except KeyError:
            pass  # Allow parent to handle exception
        return super().get(key)
