"""
This file provides the Job that will extract the information we need in order to contrast enhance the slide images.
"""

from ..job import Job

import os
import cv2
import json
import itertools
import numpy as np
from multiprocessing import Pool

from ....util.tools import get_polygon_outer_bounding_box, crop_image_from_normalized_coordinates,\
                           linear_interpolation
from ....util.histogram import get_histogram_min_max_with_percentile


def get_min_max_boundary_from_file(args):
    """
    Loads a frame from disk and computes the boundaries for the histogram stretching.

    :param args:
        A tuple (filename, rect) where
            filename: The filename to be read
            rect: The location of the slides

    The image is cropped as specified by rect, then it is converted to grayscale and
    we extract the histogram boundaries from that.
    """
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
    Extracts the min and max boundaries we use for contrast enhancing the slides.


    The inputs of the parents are expected to be the following:
    - A list of images (specified by filename) to operate on
    - The location of the slides given as a rectangle: (x, y, widht, height)

    The output of this Job are two functions with signature :: time -> boundary.
    The first function specifies the min boundary at time t.
    The second function specifies the max boundary at time t.
    """

    name = 'contrast_enhancement_boundaries'

    # @todo(Stephan): what needs to be serialized
    attributes_to_serialize = ['min_bounds',
                               'max_bounds']

    def __init__(self,
                 *args,
                 **kwargs):
        super(ContrastEnhancementBoundaries, self).__init__(*args, **kwargs)

        self._get_previous_boundaries()

    def _get_previous_boundaries(self):
        if not os.path.exists(self.json_filename):
            return None

        with open(self.json_filename) as f:
            d = json.load(f)

            for key in self.attributes_to_serialize:

                if key in d:
                    setattr(self, key, d[key])


    def run(self, *args, **kwargs):
        assert(len(args) >= 3)

        # First parent is ffmpeg
        image_list = args[0]

        # Second parent is selected slide
        slide_crop_rect = get_polygon_outer_bounding_box(args[1])

        # Third parent is the SegmentComputation

        pool = Pool(processes=6)

        boundaries = pool.map(get_min_max_boundary_from_file, itertools.izip(image_list, itertools.repeat(slide_crop_rect)))

        # Create two single lists
        self.min_bounds, self.max_bounds = zip(*boundaries)

        self.serialize_state()


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
                self.boundary_for_segment = map(self.get_histogram_boundary_for_segment, self.segments)
                return

            def __call__(self, t):
                segment_index = 0
                for (start,end) in self.segments:

                    if (start <= t) and (t <= end):
                        # We are inside a segment and thus know the boundaries
                        return self.boundary_for_segment[segment_index]

                    elif (t < start):
                        if segment_index == 0:
                            # @note(Stephan):
                            # In this case, we are before the first segment, return the default boundary.
                            return self.default_boundary

                        else:
                            # We are between two segments and thus have to interpolate
                            t0 = self.segments[segment_index - 1][1]   # End of last segment
                            t1 = self.segments[segment_index][0]       # Start of new segment

                            boundary0 = self.boundary_for_segment[segment_index - 1]
                            boundary1 = self.boundary_for_segment[segment_index]

                            lerped_boundary = linear_interpolation(t, t0, t1, boundary0, boundary1)

                            return lerped_boundary

                    segment_index += 1

                # We are behind the last computed segment, since we have no end value to
                # interpolate, we just return the bounds of the last computed segment
                return self.boundary_for_segment[-1]

            def get_histogram_boundary_for_segment(self, segment):
                """Returns the histogram boundary for a whole segment by taking
                   the average of each boundary contained in this segment."""
                start, end = segment

                bounds_in_segment = self.boundaries[int(start):int(end)]

                boundary_sum = sum(bounds_in_segment)
                n_boundaries = len(bounds_in_segment)

                return boundary_sum / n_boundaries

        return BoundsFromTime(self.min_bounds, segments, 0), BoundsFromTime(self.max_bounds, segments, 255)


if __name__ == '__main__':
    import logging
    FORMAT = '[%(asctime)-15s] %(message)s'
    logging.basicConfig(format=FORMAT)

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    root_folder = os.path.join(os.path.dirname(__file__),
                               os.pardir,
                               os.pardir,
                               os.pardir,
                               os.pardir)

    video_folder = os.path.join(root_folder, 'Videos')
    current_video = os.path.join(video_folder, 'video_7.mp4')
    proc_folder = os.path.abspath(os.path.join(root_folder, 'tmp'))

    if not os.path.exists(proc_folder):
        os.makedirs(proc_folder)

    from .histogram_computation import HistogramsLABDiff, GatherSelections, SelectSlide
    from .ffmpeg_to_thumbnails import FFMpegThumbnailsJob
    from .histogram_correlations import HistogramCorrelationJob
    from .segment_computation import SegmentComputationJob

    HistogramsLABDiff.add_parent(GatherSelections)
    HistogramsLABDiff.add_parent(FFMpegThumbnailsJob)

    HistogramCorrelationJob.add_parent(HistogramsLABDiff)

    SegmentComputationJob.add_parent(HistogramCorrelationJob)

    ContrastEnhancementBoundaries.add_parent(FFMpegThumbnailsJob)
    ContrastEnhancementBoundaries.add_parent(SelectSlide)
    ContrastEnhancementBoundaries.add_parent(SegmentComputationJob)

    # import ipdb
    d = {'video_filename': current_video,
         'thumbnails_location': os.path.join(proc_folder, 'processing_video_7_thumbnails'),
         'json_prefix': os.path.join(proc_folder, 'processing_video_7_'),
         'segment_computation_tolerance': 0.05,
         'segment_computation_min_length_in_seconds': 2}

    boundary_job = ContrastEnhancementBoundaries(**d)
    boundary_job.process()

