"""
Contrast Enhancement Boundaries
===============================

This module provides the Job for computing the boundaries for the histogram stretching
we apply in order to enhance the contrast of the slides.

.. autosummary::

  BoundariesConvolutionOnStableSegments
  ContrastEnhancementBoundaries
  ComputeExtremaOnAndBetweenSegments
  ComputesExtremaByLinearInterpolationOnSegments

"""

from ..job import Job

import itertools
from multiprocessing import Pool

from ....util.tools import get_polygon_outer_bounding_box, crop_image_from_normalized_coordinates, \
    linear_interpolation, sort_dictionary_by_integer_key
from ....util.histogram import get_histogram_min_max_with_percentile


import numpy as np
import math


def _get_min_max_boundary_from_file(args):
    """
    Load a frame from disk and computes the boundaries for the histogram stretching.

    :param args:
        A tuple (filename, rect) where
            * filename: The filename to be read
            * rect: The location of the slides specified by normalized coordinates [x,y,width,height]

    .. note:: The histogram is computed on the cropped grayscale image.
    """

    import cv2

    filename, slide_crop_rect, percentile = args
    im = cv2.imread(filename)
    im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    slide = crop_image_from_normalized_coordinates(im_gray, slide_crop_rect)
    slidehist = cv2.calcHist([slide], [0], None, [256], [0, 256])

    boundaries = get_histogram_min_max_with_percentile(slidehist, False, percentile=percentile)

    return boundaries


class ComputeExtremaOnAndBetweenSegments(object):
    """Callable object for computing the extremas over time for each segment by averaging.

    The object first computes a unique value over each segment by taking the average of the values in
    the associated lists.
    It then acts as a continuous function (callable object) returning values of extremum as a function of time.

    Between segments, a linear interpolation is performed.

    .. note:: the average function may be replaced by any other function returning a pair of appropriate values
       representing a linear slope. The function will then interpolate on this slope.
    """

    def __init__(self, boundaries, segments, default_boundary):
        self.boundaries = boundaries
        self.segments = segments
        self.default_boundary = default_boundary

        # change here to have another effect (see get_histogram_boundaries_linear_interpolation_for_segment)
        self.boundary_for_segment = map(self.get_histogram_boundary_average_for_segment,
                                        self.segments)

        return

    def __call__(self, t):
        segment_index = 0
        for (start, end) in self.segments:

            if (start <= t) and (t <= end):
                # We are inside a segment and thus know the boundaries
                segment_begin_value, segment_end_value = self.boundary_for_segment[segment_index]

                t0 = self.segments[segment_index][0]  # Start of this segment
                t1 = self.segments[segment_index][1]  # End of this segment

                return linear_interpolation(t, t0, t1, segment_begin_value, segment_end_value)

            elif (t < start):
                if segment_index == 0:
                    # In this case, we are before the first segment
                    # Return the default boundary.
                    return self.default_boundary

                else:
                    # We are between two segments and thus have to interpolate
                    t0 = self.segments[segment_index - 1][1]  # End of last segment
                    t1 = self.segments[segment_index][0]  # Start of new segment

                    boundary0 = self.boundary_for_segment[segment_index - 1][1]  # value at the end of the segment
                    boundary1 = self.boundary_for_segment[segment_index][0]  # value at the beginning of the segment

                    lerped_boundary = linear_interpolation(t, t0, t1, boundary0, boundary1)

                    return lerped_boundary

            segment_index += 1

        # We are behind the last computed segment, since we have no end value to
        # interpolate, we just return the bounds of the last computed segment
        return self.boundary_for_segment[-1]

    def get_histogram_boundary_average_for_segment(self, segment):
        """
        Return the histogram boundary for a whole segment by taking
        the average of each boundary contained in this segment.
        """
        start, end = segment

        bounds_in_segment = self.boundaries[int(start):int(end)]

        boundary_sum = sum(bounds_in_segment)
        n_boundaries = len(bounds_in_segment)

        return [boundary_sum / n_boundaries, boundary_sum / n_boundaries]

    def get_histogram_boundaries_linear_interpolation_for_segment(self, segment):
        """
        Return the histogram boundary for a whole segment by taking
        the average of each boundary contained in this segment.
        """
        start, end = segment

        bounds_in_segment = self.boundaries[int(start):int(end)]

        # note: the initial intention of this code was to have a good slope for the
        # content of the segment. The following code is hence buggy, see the convolution class
        # below for an example of proper implementation.
        yinterp = np.interp([start, end], range(int(start), int(end)), bounds_in_segment)

        return yinterp


class ContrastEnhancementBoundaries(Job):
    """
    Job for extracting the min and max boundaries we use for enhancing the contrast of
    the slides.

    The boundaries are computed on the thumbnail image transformed to grayscale.

    .. rubric:: Runtime parameters

    * `histogram_contrast_enhancement_percentile` the percentile (in `[0, 1]`) of the
      histogram that defines the lower or upper bound. Defaults to `0.01 (1%)`

    .. rubric:: Workflow inputs

    The inputs of the parents are expected to be the following:

    * A list of images (specified by filename) to operate on
    * The location of the slides given as a rectangle: [x, y, widht, height]
    * A list of stable segments `[t_segment_start, t_segment_end]`

    .. rubric:: Workflow outputs

    The output of this Job are two functions with signature::

        time -> boundary.

    * The first function specifies the min boundary at time t.
    * The second function specifies the max boundary at time t.

    .. note::

        These functions check if the specified time lies in a stable segment.
        If it does, it returns the average boundaries for this segment.
        If t does not belong to a stable segment we interpolate linearly
        between the boundaries of the two segments it lies between.

    .. rubric:: Complexity

    Linear in the number of thumbnails. The thumbnails are read once.
    """

    #: Name of the job in the workflow
    name = 'contrast_enhancement_boundaries'

    #: Cached inputs:
    #:
    #: * ``histogram_contrast_enhancement_percentile`` the percentile to keep for computing the boundaries from the histograms
    attributes_to_serialize = ['histogram_contrast_enhancement_percentile']

    #: Cached output:
    #:
    #: * ``min_bounds`` min sequence function of time
    #: * ``max_bounds`` max sequence function of time
    outputs_to_cache = ['min_bounds',
                        'max_bounds']

    def __init__(self,
                 *args,
                 **kwargs):
        super(ContrastEnhancementBoundaries, self).__init__(*args, **kwargs)

        self.histogram_contrast_enhancement_percentile = float(kwargs['histogram_contrast_enhancement_percentile']) if 'histogram_contrast_enhancement_percentile' in kwargs else 0.01

    def run(self, *args, **kwargs):
        assert(len(args) >= 2)

        # First parent is ffmpeg (list of thumbnails)
        image_list = args[0]

        # Second parent is selected slide
        slide_crop_rect = get_polygon_outer_bounding_box(args[1])

        pool = Pool(processes=6)

        boundaries = pool.map(_get_min_max_boundary_from_file,
                              itertools.izip(image_list,
                                             itertools.repeat(slide_crop_rect),
                                             itertools.repeat(self.histogram_contrast_enhancement_percentile)))

        # Create two single lists
        self.min_bounds, self.max_bounds = map(list, zip(*boundaries))

    def get_outputs(self):
        super(ContrastEnhancementBoundaries, self).get_outputs()

        if (self.min_bounds is None) or (self.max_bounds is None):
            raise RuntimeError('The histogram boundaries for contrast enhancement have not been computed yet.')

        return self.min_bounds, self.max_bounds


class ComputesExtremaByLinearInterpolationOnSegments(object):
    """Callable object for computing the extremas over time from segments list of values.

    The segment list of values should represent an estimation of the extremas every seconds (sampling time
    for the estimation of the extreams). The callable object performs a linear interpolation of those values
    within segments to have a continuous function of time.

    Between segments, a linear interpolation is also performed. """

    def __init__(self, boundaries, segments, default_boundary):
        self.boundaries = boundaries
        self.segments = segments
        self.default_boundary = default_boundary
        return

    def __call__(self, t):
        segment_index = 0

        for (start, end) in self.segments:

            if (start <= t) and (t <= end - 1):

                # We are inside a segment and thus know the boundaries
                current_segment_values = self.boundaries[segment_index]

                t0 = int(math.floor(t))  # linear interpolation between each of the elements
                t1 = t0 + 1  # spaced by 1 sec

                if (t == end - 1):
                    assert(current_segment_values[int(t) - int(start)] == current_segment_values[-1])
                    return current_segment_values[int(t) - int(start)]

                return linear_interpolation(t,
                                            t0, t1,
                                            current_segment_values[t0 - int(start)], current_segment_values[t1 - int(start)])

            elif (t < start):
                if segment_index == 0:
                    # In this case, we are before the first segment
                    # Return the default boundary.
                    return self.default_boundary

                else:
                    # We are between two segments and thus have to interpolate
                    t0 = self.segments[segment_index - 1][1]  # End of last segment
                    t1 = self.segments[segment_index][0]  # Start of new segment

                    boundary0 = self.boundaries[segment_index - 1][-1]  # value at the end of the segment
                    boundary1 = self.boundaries[segment_index][0]  # value at the beginning of the segment

                    lerped_boundary = linear_interpolation(t, t0, t1, boundary0, boundary1)

                    return lerped_boundary

            segment_index += 1

        # We are behind the last computed segment, since we have no end value to
        # interpolate, we just return the bounds of the last computed segment
        return self.boundaries[max(self.boundaries.keys())][-1]


class BoundariesConvolutionOnStableSegments(Job):
    """Post processing of the slides boundaries for enhancing the contrast of
    the slides.

    This particular Job performs a convolution with some 'kernel'
    (currently only a box kernel), and a pair of callable object representing functions of
    the min/max wrt. time (see :py:class:`ComputesExtremaByLinearInterpolationOnSegments`).

    .. rubric:: Runtime parameters

    * `size_average_window` is the size of the kernel used for averaging. This size
      is expressed in the same unit of /time/ as the frames generated by the
      boundaries (histogram percentile).

    .. rubric:: Workflow inputs

    The inputs of the parents are expected to be the following:

    * a 2-uple where each element is a list. Each list contains respectively the min and max
      computed from the histograms (at some frequency, eg. 1 value per second).
    * A list of stable segments `[t_segment_start, t_segment_end]`

    .. rubric:: Example

    An example of processing with convolutions within segments (and linear approximation for small segments)
    can be seen in the following image

      .. image:: /_static/grey_level_smoothing_over_time.png

    """

    #: Name of the job in the workflow
    name = 'moving_average_on_stable_segments'

    #: Cached inputs:
    #:
    #: * ``histogram_contrast_enhancement_percentile`` the percentile to keep for computing the boundaries from the histograms
    attributes_to_serialize = ['size_average_window']

    #: Cached output:
    #:
    #: * ``min_bounds_averaged`` min sequence function of time
    #: * ``max_bounds_averaged`` max sequence function of time
    outputs_to_cache = ['min_bounds_averaged',
                        'max_bounds_averaged']

    def __init__(self,
                 *args,
                 **kwargs):
        super(BoundariesConvolutionOnStableSegments, self).__init__(*args, **kwargs)

        # this is in seconds
        self.size_average_window = float(kwargs['size_average_window']) if 'size_average_window' in kwargs else 120
        self.size_average_window |= 1

    def load_state(self):
        """
        Sort the histograms by ``segment index`` in order to be able to compare states.

        This is necessary because the json module can load and store dictionaries
        out of order (and with string keys).
        """
        state = super(BoundariesConvolutionOnStableSegments, self).load_state()

        if state is None:
            return None

        min_bounds_averaged = state['min_bounds_averaged']
        min_bounds_averaged = sort_dictionary_by_integer_key(min_bounds_averaged)
        state['min_bounds_averaged'] = min_bounds_averaged

        max_bounds_averaged = state['max_bounds_averaged']
        max_bounds_averaged = sort_dictionary_by_integer_key(max_bounds_averaged)
        state['max_bounds_averaged'] = max_bounds_averaged

        return state

    def run(self, *args, **kwargs):
        assert(len(args) >= 2)

        # First parent is the computed boundaries
        min_bounds, max_bounds = args[0]

        kernel = np.asarray([1. / self.size_average_window] * self.size_average_window, dtype=np.double)

        # second parent is the computed stable segments
        segments = args[1]

        min_segments = {}
        max_segments = {}

        for segment_index, s in enumerate(segments):
            start = int(s[0])
            stop = int(s[1])
            current_min = min_bounds[start:stop]
            current_max = max_bounds[start:stop]

            if len(current_min) > self.size_average_window:
                current_min_conv = np.convolve(current_min, kernel, 'same')
                current_max_conv = np.convolve(current_max, kernel, 'same')

                # we replicate the values on the location where the kernels do
                # not overlap fully.
                mid_point = len(kernel) // 2
                current_min_conv[0:mid_point] = current_min_conv[mid_point]
                current_min_conv[-mid_point:] = current_min_conv[-mid_point - 1]

                current_max_conv[0:mid_point] = current_max_conv[mid_point]
                current_max_conv[-mid_point:] = current_max_conv[-mid_point - 1]

                regularized_min = current_min_conv
                regularized_max = current_max_conv
            else:
                x = np.array(xrange(start, stop), dtype=np.double).reshape(-1, 1)
                obs = np.hstack((x, np.ones(stop - start).reshape(-1, 1)))

                min_reg = np.linalg.lstsq(obs, current_min)[0]
                max_reg = np.linalg.lstsq(obs, current_max)[0]

                regularized_min = (min_reg.T * obs).sum(axis=1)
                regularized_max = (max_reg.T * obs).sum(axis=1)

            # to regular python lists
            min_segments[segment_index] = regularized_min.tolist()
            max_segments[segment_index] = regularized_max.tolist()

        # Create two single lists
        self.min_bounds_averaged, self.max_bounds_averaged = min_segments, max_segments

    def get_outputs(self):
        super(BoundariesConvolutionOnStableSegments, self).get_outputs()

        if (self.min_bounds_averaged is None) or (self.max_bounds_averaged is None):
            raise RuntimeError('The post-processed boundaries for contrast enhancement have not been computed yet.')

        segments = self.compute_segments.get_outputs()

        return ComputesExtremaByLinearInterpolationOnSegments(self.min_bounds_averaged, segments, 0), \
            ComputesExtremaByLinearInterpolationOnSegments(self.max_bounds_averaged, segments, 255)
