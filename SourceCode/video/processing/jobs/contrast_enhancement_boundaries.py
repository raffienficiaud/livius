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
    attributes_to_serialize = ['boundaries']

    def __init__(self,
                 *arg,
                 **kwargs):
        super(ContrastEnhancementBoundaries, self).__init__(*args, **kwargs)


    def run(self, *args, **kwargs):
        """Computes the Lab space of the image and the histogram_boundaries for the slide enhancement."""

        slide_crop_coordinates = get_polygon_outer_bounding_box(args[0])
        image_list = args[1]

        self.boundaries = {}

        for index, filename in enumerate(image_list):

            im = cv2.imread(file_name)
            im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

            resized_y, resized_x = im_gray.shape

            min_y = slide_crop_coordinates[0] * resized_y
            max_y = slide_crop_coordinates[1] * resized_y
            min_x = slide_crop_coordinates[2] * resized_x
            max_x = slide_crop_coordinates[3] * resized_x

            slide = im_gray[min_y: max_y, min_x: max_x]
            slidehist = cv2.calcHist([slide], [0], None, [256], [0, 256])

            min_boundary, max_boundary = get_histogram_min_max_with_percentile(slidehist, False)

            self.boundaries['min'][index] = min_boundary
            self.boundaries['max'][index] = max_boundary


        self.serialize_state()



    def get_outputs(self):
        super(ContrastEnhancementBoundaries, self).get_outputs()

        pass



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
    current_video = os.path.join(video_folder, 'Video_7.mp4')
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







