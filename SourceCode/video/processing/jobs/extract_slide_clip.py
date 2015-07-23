"""
This file provides the Job for extracting the slide clip from the videofile
"""

from ..job import Job

import cv2
import numpy as np
import os

from .histogram_computation import SelectSlide
from ....util.tools import get_transformation_points_from_normalized_rect, get_polygon_outer_bounding_box

class SlideWarpJob(Job):
    name = 'warp_slides'
    attributes_to_serialize = []
    parents = [SelectSlide]

    def __init__(self, *args, **kwargs):
        super(SlideWarpJob, self).__init__(*args, **kwargs)

        assert('slide_clip_desired_format' in kwargs)

    def run(self, *args, **kwargs):
        pass

    def get_outputs(self):
        super(SlideWarpJob, self).get_outputs()

        slide_location = self.select_slides.get_outputs()
        slide_rect = get_polygon_outer_bounding_box(slide_location)

        class Warper:
            """Callable object for warping the frame into perspective and cropping out the slides"""
            def __init__(self, slide_rect, desiredScreenLayout):
                self.slide_rect = slide_rect
                self.desiredScreenLayout = desiredScreenLayout

            def __call__(self, image):
                """Cuts out the slides from the video and warps them into perspective.
                """
                # Extract Slides
                slideShow = np.array([[0,0],
                                      [self.desiredScreenLayout[0]-1, 0],
                                      [self.desiredScreenLayout[0]-1, self.desiredScreenLayout[1]-1],
                                      [0,self.desiredScreenLayout[1]-1]],
                                     np.float32)

                slide_coordinates = get_transformation_points_from_normalized_rect(slide_rect, image)
                retval = cv2.getPerspectiveTransform(slide_coordinates, slideShow)
                warp = cv2.warpPerspective(image, retval, self.desiredScreenLayout)

                # Return slide image
                return warp

        return Warper(slide_rect, self.slide_clip_desired_format)


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
    from .contrast_enhancement_boundaries import ContrastEnhancementBoundaries

    HistogramsLABDiff.add_parent(GatherSelections)
    HistogramsLABDiff.add_parent(FFMpegThumbnailsJob)

    HistogramCorrelationJob.add_parent(HistogramsLABDiff)

    SegmentComputationJob.add_parent(HistogramCorrelationJob)

    ContrastEnhancementBoundaries.add_parent(FFMpegThumbnailsJob)
    ContrastEnhancementBoundaries.add_parent(SelectSlide)
    ContrastEnhancementBoundaries.add_parent(SegmentComputationJob)

    SlideWarpJob.add_parent(SelectSlide)

    # import ipdb
    d = {'video_filename': current_video,
         'thumbnails_location': os.path.join(proc_folder, 'processing_video_7_thumbnails'),
         'json_prefix': os.path.join(proc_folder, 'processing_video_7_'),
         'segment_computation_tolerance': 0.05,
         'segment_computation_min_length_in_seconds': 2,
         'slide_clip_desired_format': [1280, 960]}

    boundary_job = SlideWarpJob(**d)
    boundary_job.process()


