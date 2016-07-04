"""
Functor
=======

This module provides a class that wraps data into a callable object.
"""


class Functor(object):
    """
    Creates a callable object from the given data.
    The data must be indexable by the arguments that are provided when calling the object.

    :param data: The data which is accessed by the arguments given when
                  this object is called.
    :type data: Anything with a specified ``__getitem__`` method

    :param transform: An additional function that transforms the indexed item (e.g turns it into a np.array)

    """

    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __call__(self, *args):
        """Index the stored data.

        :param args: One or more indices we want to access the data with.

        .. note::

           The arguments when calling the object must be provided in order of indexing
        """
        item = None

        for index in args:
            if item is None:
                item = self.data[index]
            else:
                item = item[index]

        if self.transform is not None:
            item = self.transform(item)

        return item
