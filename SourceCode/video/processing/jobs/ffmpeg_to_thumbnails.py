
"""Defines the FFMpeg job that transforms a video file into a sequence of thumbnail images.
Those thumbnails are more suitable for analysis."""

import os
import subprocess
import time
from ..job import Job


def extract_thumbnails(video_file_name, output_width, output_folder):
    """Extracts the thumbnails using FFMpeg.

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

    name = "thumbnails"
    attributes_to_serialize = ['video_filename',
                               'video_fps',
                               'video_width',
                               'output_location',
                               'output_files']

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
                 video_filename,
                 video_width=None,
                 video_fps=None,
                 output_location=None,
                 *args,
                 **kwargs):
        """
        :param output_location: absolute location of the generated thumbnails
        """

        super(FFMpegThumbnailsJob, self).__init__(*args, **kwargs)

        if video_filename is None:
            raise RuntimeError("The video file name cannot be empty")

        # this is necessary because the json files are stored in unicode, and the
        # comparison of the list of files should work (unicode path operations
        # is unicode)
        video_filename = unicode(video_filename)

        if video_width is None:
            video_width = 640

        if video_fps is None:
            video_fps = 1

        if output_location is None:
            output_location = self.get_thumbnail_location(video_filename)

        self.video_filename = os.path.abspath(video_filename)
        self.video_fps = video_fps
        self.video_width = video_width
        self.output_location = output_location
        self.json_filename = os.path.splitext(video_filename)[0] + '_' + self.name + '.json'

        # read back the output files if any
        self.output_files = self._get_files()

    def is_up_to_date(self):
        """Returns False if the state of the json dump is not the same as
        the current state of the instance. This value is indicated for all
        parents from this instance up to the root"""
        if not os.path.exists(self.output_location):
            return False

        if not self.output_files:
            return False

        return super(FFMpegThumbnailsJob, self).is_up_to_date()

    def run(self, *args, **kwargs):

        if self.is_up_to_date():
            return True

        if not os.path.exists(self.output_location):
            os.makedirs(self.output_location)

        extract_thumbnails(video_file_name=self.video_filename,
                           output_width=self.video_width,
                           output_folder=self.output_location)

        # save the output files
        self.output_files = self._get_files()

        # commit to the json dump
        self.serialize_state()

        return True

    def _get_files(self):
        if not os.path.exists(self.output_location):
            return []

        possible_output = [os.path.abspath(os.path.join(self.output_location, i)) for i in os.listdir(self.output_location) if i.find('frame-') != -1]
        possible_output.sort()

        return possible_output

    def get_outputs(self):
        super(FFMpegThumbnailsJob, self).get_outputs()
        return self._get_files()


def factory():
    return FFMpegThumbnailsJob
