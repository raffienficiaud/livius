"""
This file provides a class that wraps data into a callable object.
"""

class Functor(object):
    """
    Creates a callable object from the given data.
    The data must be indexable by the arguments that are provided when calling the object.

    :note:
        The arguments when calling the object must be provided in order of indexing
    """

    def __init__(self, data, transform=None):
        """
        :param (array or dictionary) data: The data which is accessed by the arguments given when
                                           this object is called.
        :param transform: An additional function that transforms the indexed item (e.g turns it into a np.array)
        """
        self.data = data
        self.transform = transform

    def __call__(self, *args):
        item = None

        for index in args:
            if item is None:
                item = self.data[index]
            else:
                item = item[index]

        if self.transform is not None:
            item = self.transform(item)

        return item

