"""
This file provides the Job that will extract the information we need in order to contrast enhance the slide images.
"""

from ..job import Job

import os
import cv2
import numpy as np

from ....util.tools import get_polygon_outer_bounding_box, crop_image_from_normalized_coordinates


class ContrastEnhancementBoundaries(Job):
    """
    Extracts the min and max boundaries we use for contrast enhancing the slides.
    """

    name = 'contrast_enhancement_boundaries'

    # @todo(Stephan): what needs to be serialized
    attributes_to_serialize = ['min_bounds',
                               'max_bounds']

    def __init__(self,
                 *args,
                 **kwargs):
        super(ContrastEnhancementBoundaries, self).__init__(*args, **kwargs)

        # @todo(Stephan): Read previous state



    # @todo(Stephan): Multiprocess this
    def run(self, *args, **kwargs):
        """Computes the Lab space of the image and the histogram_boundaries for the slide enhancement."""

        # First parent is ffmpeg
        image_list = args[0]

        # Second parent is selected slide
        slide_crop_rect = get_polygon_outer_bounding_box(args[1])

        # @todo(Stephan): SegmentCompuation needs to be a parent!

        self.min_bounds = []
        self.max_bounds = []

        for index, filename in enumerate(image_list):

            im = cv2.imread(file_name)
            im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

            resized_y, resized_x = im_gray.shape

            slide = crop_image_from_normalized_coordinates(im_gray, slide_crop_rect)
            slidehist = cv2.calcHist([slide], [0], None, [256], [0, 256])

            min_boundary, max_boundary = get_histogram_min_max_with_percentile(slidehist, False)

            self.min_bounds.append(min_boundary)
            self.max_bounds.append(max_boundary)


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







