"""
FFMpeg To Thumbnails
====================

This module defines the FFMpeg job that transforms a video file into a sequence of thumbnail images.

Those thumbnails are more suitable for analysis.
"""

import os
import subprocess
import time
from ..job import Job


def extract_thumbnails(video_file_name, output_width, output_folder):
    """Extract the thumbnails using FFMpeg.

    :param video_file_name: name of the video file to process
    :param output_width: width of the resized images
    :param output_folder: folder where the thumbnails are stored

    """
    args = ['ffmpeg',
            '-i', os.path.abspath(video_file_name),
            '-r', '1',
            '-vf', 'scale=%d:-1' % output_width,
            '-f', 'image2', '%s/frame-%%05d.png' % os.path.abspath(output_folder)]
    proc = subprocess.Popen(args)

    return_code = proc.poll()
    while return_code is None:
        time.sleep(1)
        return_code = proc.poll()

    return


class FFMpegThumbnailsJob(Job):

    """
    Job for extracting thumbnails from a video.

    **Job Parameters**

    Parameters of the Job (expected to be passed when constructing a workflow instance):

    :param video_filename: The name of the video file to process

    Optional parameters:

    :param thumbnails_location: absolute location of the generated thumbnails
    :param video_width: The width of the generated thumbnails
    :param video_fps: How many frames per second to extract

    **Parent inputs**

    None

    **Job outputs**

    This job returns
        * A list of filenames that specify the generated thumbnails.

    """

    name = "ffmpeg_thumbnails"
    #:
    attributes_to_serialize = ['video_filename',
                               'video_fps',
                               'video_width',
                               'thumbnails_location']
    #:
    outputs_to_cache = ['thumbnail_files']

    @staticmethod
    def get_thumbnail_root(video_filename):
        """Indicates the root where files are stored"""
        return os.path.dirname(video_filename)

    @staticmethod
    def get_thumbnail_location(video_filename):
        return os.path.join(FFMpegThumbnailsJob.get_thumbnail_root(video_filename),
                            'thumbnails',
                            os.path.splitext(os.path.basename(video_filename))[0])

    def __init__(self,
                 *args,
                 **kwargs):
        """
        Expects the following named arguments in kwargs:
        :param video_filename: The name of the video file to process

        Optional parameters:
        :param thumbnails_location: absolute location of the generated thumbnails
        :param video_width: The width of the generated thumbnails
        :param video_fps: How many frames per second to extract
        """
        super(FFMpegThumbnailsJob, self).__init__(*args, **kwargs)

        video_filename = kwargs.get('video_filename', None)
        if video_filename is None:
            raise RuntimeError("The video file name cannot be empty")
        if not os.path.exists(video_filename):
            raise RuntimeError("The video file %s does not exist" % os.path.abspath(video_filename))

        # this is necessary because the json files are stored in unicode, and the
        # comparison of the list of files should work (unicode path operations
        # is unicode)
        video_filename = unicode(video_filename)

        # Put in default values if they are not passed in the kwargs
        video_width = kwargs.get('video_width', 640)
        video_fps = kwargs.get('video_fps', 1)
        thumbnails_location = kwargs.get('thumbnails_location', self.get_thumbnail_location(video_filename))

        self.video_filename = os.path.abspath(video_filename)
        self.video_fps = video_fps
        self.video_width = video_width
        self.thumbnails_location = os.path.abspath(unicode(thumbnails_location))  # same issue as for the video filename

    def run(self, *args, **kwargs):

        if self.is_up_to_date():
            return True

        if not os.path.exists(self.thumbnails_location):
            os.makedirs(self.thumbnails_location)

        extract_thumbnails(video_file_name=self.video_filename,
                           output_width=self.video_width,
                           output_folder=self.thumbnails_location)

        # save the output files
        self.thumbnail_files = self._get_files()

    def _get_files(self):
        if not os.path.exists(self.thumbnails_location):
            return []

        possible_output = [os.path.abspath(os.path.join(self.thumbnails_location, i)) for i in os.listdir(self.thumbnails_location) if i.find('frame-') != -1]
        possible_output.sort()

        return possible_output

    def get_outputs(self):
        super(FFMpegThumbnailsJob, self).get_outputs()
        return self._get_files()


def factory():
    return FFMpegThumbnailsJob


class NumberOfFilesJob(Job):

    """Job that stores how many thumbnails were generated by the ffmpeg Job."""

    name = 'number_of_files'
    parents = [FFMpegThumbnailsJob]
    outputs_to_cache = ['nb_files']

    def __init__(self, *args, **kwargs):
        super(NumberOfFilesJob, self).__init__(*args, **kwargs)

    def run(self, *args, **kwargs):
        filelist = args[0]
        self.nb_files = len(filelist)

    def get_outputs(self):
        super(NumberOfFilesJob, self).get_outputs()

        return self.nb_files
