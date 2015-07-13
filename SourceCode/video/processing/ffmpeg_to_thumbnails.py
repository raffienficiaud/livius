
"""Defines the FFMpeg task"""

import os
import subprocess
import time
from .job import Job


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
    attributes_to_serialize = ['video_filename', 'video_fps', 'video_width', 'output_location']

    @staticmethod
    def get_thumbnail_location(video_filename):
        return os.path.join(os.path.dirname(video_filename),
                            'thumbnails',
                            os.path.splitext(os.path.basename(video_filename))[0])

    def __init__(self,
                 video_filename,
                 video_width = None,
                 video_fps = None,
                 output_location = None,
                 *args, 
                 **kwargs):
        """
        :param output_location: absolute location of the generated thumbnails
        """

        super(FFMpegThumbnailsJob, self).__init__(*args, **kwargs)

        if video_filename is None:
            raise RuntimeError("The video file name cannot be empty")

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

    def is_up_to_date(self):
        """Returns False if the state of the json dump is not the same as
        the current state of the instance. This value is indicated for all
        parents from this instance up to the root"""
        if not os.path.exists(self.output_location):
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

        # commit to the json dump
        self.serialize_state()

        return True

    def get_outputs(self):
        super(FFMpegThumbnailsJob, self).get_outputs()

        possible_output = [os.path.abspath(os.path.join(self.output_location, i)) for i in os.listdir(self.output_location) if i.find('frame-') != -1]
        possible_output.sort()

        return possible_output


def factory():
    return FFMpegThumbnailsJob
