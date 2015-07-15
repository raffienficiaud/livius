'''
Defines a job for selecting the slides location from a video file name
'''

import os
import json
import cv2
from random import randint

from ..job import Job
from ....util.user_interaction import get_polygon_from_user


class SelectSlideJob(Job):
    """This Job shows one frame of the input video (or extracted thumbnail image) to the user and asks
    for a polygon defining the location of the slides"""

    name = "slide_location"
    attributes_to_serialize = ['video_filename',
                               'points']

    def __init__(self,
                 video_filename,
                 *args,
                 **kwargs):
        """
        :param video_filename: the video file
        """

        # the video_filename is for being able to pass this parameter
        # to a potential parent class (in this case potentially consummed by
        # the FFMpeg)
        super(SelectSlideJob, self).__init__(video_filename=video_filename,
                                             *args, **kwargs)
        # video_filename = kwargs.get('video_filename', None)

        if video_filename is None:
            raise RuntimeError("The video file name cannot be empty")
        if not os.path.exists(video_filename):
            raise RuntimeError("The video file %s does not exist" % os.path.abspath(video_filename))

        # this is necessary because the json files are stored in unicode, and the
        # comparison of the list of files should work (unicode path operations
        # is unicode)
        video_filename = unicode(video_filename)

        self.video_filename = os.path.abspath(video_filename)

        # read back the output files if any
        self.points = self._get_points()

    def _get_points(self):
        if not os.path.exists(self.json_filename):
            return None

        with open(self.json_filename) as f:
            d = json.load(f)
            if 'points' not in d:
                return None
            return d['points']

    def is_up_to_date(self):
        """Returns False if the selection has not been done already"""
        if self.points is None:
            return False

        return super(SelectSlideJob, self).is_up_to_date()

    def run(self, *args, **kwargs):

        if self.is_up_to_date():
            return True

        if not args:
            cap = cv2.VideoCapture(self.video_filename)
            cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, 500)  # drop the first 500 frames, just like that

            width = cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)
            height = cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)

            _, im_for_selection = cap.read()
        else:
            im_for_selection = cv2.imread(args[0][randint(0, len(args[0]))])
            width, height, _ = im_for_selection.shape

        self.points = get_polygon_from_user(im_for_selection, 4, 'Select the location of the slides')
        self.points = [(float(i) / width, float(j) / height) for (i, j) in self.points]

        # commit to the json dump
        self.serialize_state()

        return

    def get_outputs(self):
        super(SelectSlideJob, self).get_outputs()
        if self.points is None:
            raise RuntimeError('The points have not been selected yet')
        return self.points


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

    from .ffmpeg_to_thumbnails import FFMpegThumbnailsJob
    SelectSlideJob.add_parent(FFMpegThumbnailsJob)

    # import ipdb
    d = {'video_filename': current_video,
         'thumbnails_location': os.path.join(proc_folder, 'processing_video_7_thumbnails'),
         'json_prefix': os.path.join(proc_folder, 'processing_video_7_')}

    job_instance = SelectSlideJob(**d)
    job_instance.process()

    # should not pop out a new window because same params
    job_instance2 = SelectSlideJob(**d)
    job_instance2.process()
