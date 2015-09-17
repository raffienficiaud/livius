"""
Workflow
========

This module defines several workflow using the standard/basic Jobs. To list
all the workflows from the command line, the ``--list-workflows`` switch may
be used.

"""

from .jobs.ffmpeg_to_thumbnails import factory as ffmpeg_factory
from .jobs.histogram_computation import HistogramsLABDiff, GatherSelections, SelectSlide
from .jobs.ffmpeg_to_thumbnails import FFMpegThumbnailsJob, NumberOfFilesJob
from .jobs.histogram_correlations import HistogramCorrelationJob
from .jobs.segment_computation import SegmentComputationJob
from .jobs.contrast_enhancement_boundaries import ContrastEnhancementBoundaries
from .jobs.extract_slide_clip import ExtractSlideClipJob

import os

import logging
FORMAT = '[%(asctime)-15s] %(message)s'
logging.basicConfig(format=FORMAT)

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

# note
# in order to appear in the --list-workflow command line option, the workflow in this file
# should start with the 'workflow_' name

def workflow_thumnails_only():
    """Return a workflow made by only one node that extracts the thumnails from a video."""
    ffmpeg = ffmpeg_factory()

    return ffmpeg

def workflow_slide_detection_window():
    """Returns a trivial workflow that requires the user to select the slide locations only"""

    return SelectSlide

def workflow_extract_slide_clip():
    """
    Return a workflow that extracs the slide clip from a video.

    Consists of many tasks such as

        * ffmpg thumbnail generation
        * Polygon Selection for the Slides and Speaker
        * Histogram Computations
        * Histogram Correlations
        * Segment Computation
        * Perspective Transformations and Contrast Enhancement.

    """
    HistogramsLABDiff.add_parent(GatherSelections)
    HistogramsLABDiff.add_parent(FFMpegThumbnailsJob)

    HistogramCorrelationJob.add_parent(HistogramsLABDiff)
    HistogramCorrelationJob.add_parent(NumberOfFilesJob)

    SegmentComputationJob.add_parent(HistogramCorrelationJob)
    SegmentComputationJob.add_parent(NumberOfFilesJob)

    ContrastEnhancementBoundaries.add_parent(FFMpegThumbnailsJob)
    ContrastEnhancementBoundaries.add_parent(SelectSlide)
    ContrastEnhancementBoundaries.add_parent(SegmentComputationJob)

    return ExtractSlideClipJob


def process(workflow_instance, **kwargs):

    instance = workflow_instance(**kwargs)
    instance.process()

    out = instance.get_outputs()
    instance.serialize_state()

    return out


if __name__ == '__main__':

    # import os
    # from tempfile import mkdtemp
    # tmpdir = mkdtemp()

    # d = dict([('video_filename',
    #           os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir, "Videos", "video_7.mp4")
    #           )])

    # current_workflow = workflow_thumnails_only()
    # outputs = process(current_workflow,
    #                   json_prefix=os.path.join(tmpdir, 'test_video7'),
    #                   **d)

    # print outputs

    import logging
    FORMAT = '[%(asctime)-15s] %(message)s'
    logging.basicConfig(format=FORMAT)

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    root_folder = os.path.join(os.path.dirname(__file__),
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

    workflow = workflow_extract_slide_clip()

    params = {'video_filename': current_video,
              'thumbnails_location': os.path.join(proc_folder, 'processing_video_7_thumbnails'),
              'json_prefix': os.path.join(proc_folder, 'processing_video_7_'),
              'segment_computation_tolerance': 0.05,
              'segment_computation_min_length_in_seconds': 2,
              'slide_clip_desired_format': [1280, 960],
              'nb_vertical_stripes': 10}

    outputs = process(workflow, **params)

    outputs.write_videofile(os.path.join(slide_clip_folder, 'slideclip.mp4'))
