"""This file provides the Job for extracting the slide clip from the videofile."""

from ..job import Job

import cv2
import numpy as np
import os

# from moviepy.editor import *
# from moviepy.Clip import *
from moviepy.editor import VideoFileClip

from .histogram_computation import SelectSlide
from .contrast_enhancement_boundaries import ContrastEnhancementBoundaries

from ....util.tools import get_transformation_points_from_normalized_rect,\
                           get_polygon_outer_bounding_box


class WarpSlideJob(Job):

    """
    Warps the slides into perspective and crops them.

    Parameters of the Job (expected to be passed when constructing a workflow instance):
        slide_clip_desired_format  The output size of the slide images for the composite video.


    Inputs of the parents:
        The location of the slides given as a list of points.
        (The slide location is then assumed to be the outer bounding box of this polygon)

    Returns a Callable object that provides a function

            f :: img -> img

    which applies the wanted slide transformation.
    """

    name = 'warp_slides'
    attributes_to_serialize = ['slide_clip_desired_format']
    parents = [SelectSlide]

    def __init__(self, *args, **kwargs):
        super(WarpSlideJob, self).__init__(*args, **kwargs)

        assert('slide_clip_desired_format' in kwargs)

    def run(self, *args, **kwargs):
        pass

    def get_outputs(self):
        super(WarpSlideJob, self).get_outputs()

        slide_location = self.select_slides.get_outputs()
        slide_rect = get_polygon_outer_bounding_box(slide_location)

        class Warper:

            """Callable object for warping the frame into perspective and cropping the slides."""

            def __init__(self, slide_rect, desiredLayout):
                self.slide_rect = slide_rect

                # @note(Stephan): Convert to tuple (List is for JSON storage)
                self.desiredLayout = (desiredLayout[0], desiredLayout[1])

            def __call__(self, image):
                """Cut out the slides from the video and warps them into perspective."""
                # Extract Slides
                slideShow = np.array([[0, 0],
                                      [self.desiredLayout[0]-1, 0],
                                      [self.desiredLayout[0]-1, self.desiredLayout[1]-1],
                                      [0, self.desiredLayout[1]-1]],
                                     np.float32)

                slide_coordinates = get_transformation_points_from_normalized_rect(slide_rect, image)
                retval = cv2.getPerspectiveTransform(slide_coordinates, slideShow)
                warp = cv2.warpPerspective(image, retval, self.desiredLayout)

                # Return slide image
                return warp

        return Warper(slide_rect, self.slide_clip_desired_format)


class EnhanceContrastJob(Job):

    """
    Enhances the contrast in the slide images.

    Inputs of the parents:
        A tuple of functions that provide the min and max boundary used for
        contrast enhancing


    Returns a callable object that provides a function:

            f :: img, t -> img

    which enhances the contrast of the given image.
    """

    name = 'enhance_contrast'
    attributes_to_serialize = []
    parents = [ContrastEnhancementBoundaries]

    def __init__(self, *args, **kwargs):
        super(EnhanceContrastJob, self).__init__(*args, **kwargs)

    def run(self, *args, **kwargs):
        pass

    def get_outputs(self):
        super(EnhanceContrastJob, self).get_outputs()

        get_min_bounds, get_max_bounds = self.contrast_enhancement_boundaries.get_outputs()

        class ContrastEnhancer:

            """Callable object for enhancing the contrast of the slides."""

            def __init__(self, get_min_bounds, get_max_bounds):
                self.get_min_bounds = get_min_bounds
                self.get_max_bounds = get_max_bounds

            def __call__(self, image, t):
                """Perform contrast enhancement by putting the colors into their full range."""
                # Retrieve histogram boundaries for this frame
                min_val = self.get_min_bounds(t)
                max_val = self.get_max_bounds(t)

                # Perform the contrast enhancement
                contrast_enhanced = 255.0 * (np.maximum(image.astype(np.float32) - min_val, 0)) / (max_val - min_val)
                contrast_enhanced = np.minimum(contrast_enhanced, 255.0)
                contrast_enhanced = contrast_enhanced.astype(np.uint8)

                return contrast_enhanced

        return ContrastEnhancer(get_min_bounds, get_max_bounds)


class ExtractSlideClipJob(Job):

    """
    Extracts the Slide Clip from the Video.

    Inputs of the parents:
        - A function for warping the Slides into perspective
        - A function for enhancing the contrast of the warped Slides


    The output is the transformed video clip.

    :note:
        The transformations are only applied at write-time.
    """

    name = 'extract_slide_clip'
    attributes_to_serialize = []
    parents = [WarpSlideJob, EnhanceContrastJob]

    def __init__(self, *args, **kwargs):
        super(ExtractSlideClipJob, self).__init__(*args, **kwargs)

    def run(self, *args, **kwargs):
        pass

    def get_outputs(self):
        super(ExtractSlideClipJob, self).get_outputs()

        warp_slide = self.warp_slides.get_outputs()
        enhance_contrast = self.enhance_contrast.get_outputs()

        clip = VideoFileClip(self.video_filename)

        def apply_effects(get_frame, t):
            """Function that chains together all the post processing effects."""
            frame = get_frame(t)

            warped = warp_slide(frame)
            contrast_enhanced = enhance_contrast(warped, t)

            return contrast_enhanced

        return clip.fl(apply_effects)


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
    slide_clip_folder = os.path.abspath(os.path.join(proc_folder, 'Slide Clip'))

    if not os.path.exists(proc_folder):
        os.makedirs(proc_folder)

    if not os.path.exists(slide_clip_folder):
        os.makedirs(slide_clip_folder)

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

    # import ipdb
    d = {'video_filename': current_video,
         'thumbnails_location': os.path.join(proc_folder, 'processing_video_7_thumbnails'),
         'json_prefix': os.path.join(proc_folder, 'processing_video_7_'),
         'segment_computation_tolerance': 0.05,
         'segment_computation_min_length_in_seconds': 2,
         'slide_clip_desired_format': [1280, 960]}

    job = ExtractSlideClipJob(**d)
    job.process()

    slideClip = job.get_outputs()

    # Writes the complete slide clip to disk
    # slideClip.write_videofile(os.path.join(slide_clip_folder, 'slideclip.mp4'))

    # Just write individual frames for testing
    slideClip.save_frame(os.path.join(slide_clip_folder, '0.png'), t=0)
    slideClip.save_frame(os.path.join(slide_clip_folder, '1.png'), t=0.5)
    slideClip.save_frame(os.path.join(slide_clip_folder, '2.png'), t=1)
    slideClip.save_frame(os.path.join(slide_clip_folder, '3.png'), t=1.5)
    slideClip.save_frame(os.path.join(slide_clip_folder, '4.png'), t=2)
    slideClip.save_frame(os.path.join(slide_clip_folder, '5.png'), t=2.5)
    slideClip.save_frame(os.path.join(slide_clip_folder, '6.png'), t=3)
    slideClip.save_frame(os.path.join(slide_clip_folder, '7.png'), t=3.5)
    slideClip.save_frame(os.path.join(slide_clip_folder, '8.png'), t=4)
