"""
Contrast Enhancement Boundaries
===============================

This module provides the Job for computing the boundaries for the histogram stretching
we apply in order to enhance the contrast of the slides.

.. autosummary::

  ContrastEnhancementBoundaries

"""

from ..job import Job

import itertools
from multiprocessing import Pool

from ....util.tools import get_polygon_outer_bounding_box, crop_image_from_normalized_coordinates, \
                           linear_interpolation
from ....util.histogram import get_histogram_min_max_with_percentile


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

    filename, slide_crop_rect = args
    im = cv2.imread(filename)
    im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    resized_y, resized_x = im_gray.shape

    slide = crop_image_from_normalized_coordinates(im_gray, slide_crop_rect)
    slidehist = cv2.calcHist([slide], [0], None, [256], [0, 256])

    boundaries = get_histogram_min_max_with_percentile(slidehist, False)

    return boundaries


class ContrastEnhancementBoundaries(Job):

    """
    Job for extracting the min and max boundaries we use for contrast enhancing the slides.

    .. rubric:: Runtime parameters

    This class does not have any runtime configuration.

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
    """

    name = 'contrast_enhancement_boundaries'

    # :
    outputs_to_cache = ['min_bounds',
                        'max_bounds']

    def __init__(self,
                 *args,
                 **kwargs):
        super(ContrastEnhancementBoundaries, self).__init__(*args, **kwargs)

    def run(self, *args, **kwargs):
        assert(len(args) >= 3)

        # First parent is ffmpeg
        image_list = args[0]

        # Second parent is selected slide
        slide_crop_rect = get_polygon_outer_bounding_box(args[1])

        # Third parent is the SegmentComputation, we only access it in get_outputs().

        pool = Pool(processes=6)

        boundaries = pool.map(_get_min_max_boundary_from_file,
                              itertools.izip(image_list, itertools.repeat(slide_crop_rect)))

        # Create two single lists
        self.min_bounds, self.max_bounds = map(list, zip(*boundaries))

    def get_outputs(self):
        super(ContrastEnhancementBoundaries, self).get_outputs()

        segments = self.compute_segments.get_outputs()

        if (self.min_bounds is None) or (self.max_bounds is None):
            raise RuntimeError('The histogram boundaries for contrast enhancement have not been computed yet.')

        class BoundsFromTime(object):
            def __init__(self, boundaries, segments, default_boundary):
                self.boundaries = boundaries
                self.segments = segments
                self.default_boundary = default_boundary
                self.boundary_for_segment = map(self.get_histogram_boundary_for_segment,
                                                self.segments)
                return

            def __call__(self, t):
                segment_index = 0
                for (start, end) in self.segments:

                    if (start <= t) and (t <= end):
                        # We are inside a segment and thus know the boundaries
                        return self.boundary_for_segment[segment_index]

                    elif (t < start):
                        if segment_index == 0:
                            # In this case, we are before the first segment
                            # Return the default boundary.
                            return self.default_boundary

                        else:
                            # We are between two segments and thus have to interpolate
                            t0 = self.segments[segment_index - 1][1]  # End of last segment
                            t1 = self.segments[segment_index][0]  # Start of new segment

                            boundary0 = self.boundary_for_segment[segment_index - 1]
                            boundary1 = self.boundary_for_segment[segment_index]

                            lerped_boundary = linear_interpolation(t, t0, t1, boundary0, boundary1)

                            return lerped_boundary

                    segment_index += 1

                # We are behind the last computed segment, since we have no end value to
                # interpolate, we just return the bounds of the last computed segment
                return self.boundary_for_segment[-1]

            def get_histogram_boundary_for_segment(self, segment):
                """
                Return the histogram boundary for a whole segment by taking
                the average of each boundary contained in this segment.
                """
                start, end = segment

                bounds_in_segment = self.boundaries[int(start):int(end)]

                boundary_sum = sum(bounds_in_segment)
                n_boundaries = len(bounds_in_segment)

                return boundary_sum / n_boundaries

        return BoundsFromTime(self.min_bounds, segments, 0), \
            BoundsFromTime(self.max_bounds, segments, 255)
