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

from ....util.tools import get_polygon_outer_bounding_box, crop_image_from_normalized_coordinates
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
        # First parent is ffmpeg
        image_list = args[0]

        # Second parent is selected slide
        slide_crop_rect = get_polygon_outer_bounding_box(args[1])

        pool = Pool(processes=6)

        boundaries = pool.map(get_min_max_boundary_from_file, itertools.izip(image_list, itertools.repeat(slide_crop_rect)))

        # Create two single lists
        self.min_bounds, self.max_bounds = zip(*result)

        self.serialize_state()


    def get_outputs(self):
        super(ContrastEnhancementBoundaries, self).get_outputs()

        if (self.min_bounds is None) or (self.max_bounds is None):
            raise RuntimeError('The histogram boundaries for contrast enhancement have not been computed yet.')

        class BoundsFromTime(object):

            def __init__(self, boundaries):
                self.bounds = boundaries
                return

            def __call__(self, t):

                # @todo(Stephan):
                # We need the segments here first because we need to determine if we interpolate or take the average
                # of the bounds
                return self.bounds[t]

        # @todo(Stephan): Really return two functions here?
        return BoundsFromTime(self.min_bounds), BoundsFromTime(self.max_bounds)


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

    from .select_polygon import SelectPolygonJob
    class SelectSlide(SelectPolygonJob):
        name = 'select_slides'

    from .ffmpeg_to_thumbnails import FFMpegThumbnailsJob
    ContrastEnhancementBoundaries.add_parent(FFMpegThumbnailsJob)
    ContrastEnhancementBoundaries.add_parent(SelectSlide)

    d = {'video_filename': current_video,
         'thumbnails_location': os.path.join(proc_folder, 'processing_video_7_thumbnails'),
         'json_prefix': os.path.join(proc_folder, 'processing_video_7_')}

    boundary_job = ContrastEnhancementBoundaries(**d)
    boundary_job.process()

