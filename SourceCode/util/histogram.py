'''
This module contains several utilities for histograms
'''
import cv2


def get_histogram_min_max_with_percentile(hist,
                                          is_normalized,
                                          percentile=None):
    """Gets the p- and (1-p)-percentile as an approximation of the boundaries
       of the histogram.

    :param (array) hist: the histogram on which the boundaries should be computed
    :param (bool) is_normalized: if False, the histogram is normalized prior to the computation
    :param (Float) percentile: the percentile above/below the min/max that should be returned. If None
        0.01 (1%) is taken.
    :returns: a tuple (min, max) value for the histogram
    """
    if not is_normalized:
        hist = cv2.normalize(hist)

    if percentile is None:
        percentile = 0.01

    t_min = 0
    t_max = 255

    min_mass = 0
    max_mass = 0

    # Integrate until we reach 1% of the mass from each direction
    while min_mass < percentile:
        min_mass += hist[t_min]
        t_min += 1

    while max_mass < percentile:
        max_mass += hist[t_max]
        t_max -= 1

    return t_min, t_max
