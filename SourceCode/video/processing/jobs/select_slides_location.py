'''
Defines a job for selecting the slides location from a video file name
'''

from ..job import Job
import os
import json
import cv2
import numpy as np
from random import randint


def get_points(im,
               window_name=None):

    if window_name is None:
        window_name = ''.join([chr(ord('a') + randint(0, 26)) for _ in range(10)])

    params = type('params', (object,), {})()
    params.current_position = None
    params.click = None
    params.click_position = None
    params.points = []

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.moveWindow(window_name, 100, 100)

    def onMouse(event, x, y, flags, param):

        param.current_position = (x, y)

        if not (flags & cv2.EVENT_FLAG_LBUTTON) and params.click:
            params.click = False
            params.points.append(params.click_position)
            print params.points
            params.click_position = None

        if flags & cv2.EVENT_FLAG_LBUTTON and params.click_position is None:
            params.click = True
            params.click_position = (x, y)

    cv2.setMouseCallback(window_name, onMouse, params)
    cv2.imshow(window_name, im)

    points = params.points
    while len(params.points) < 4:

        im_draw = np.copy(im)
        if len(points) > 1:
            for index in range(1, len(points)):
                cv2.line(im_draw, points[index - 1], points[index], (255, 0, 0))

        if len(points) > 0 and params.current_position is not None:
            cv2.line(im_draw, points[-1], params.current_position, (255, 0, 0))

        cv2.imshow(window_name, im_draw)
        _ = cv2.waitKey(10)

    cv2.destroyWindow(window_name)

    return points


class SelectSlideJob(Job):

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

            _, im0_not_resized = cap.read()
        else:
            im0_not_resized = cv2.imread(args[0][randint(0, len(args[0]))])
            width, height, _ = im0_not_resized.shape

        self.points = get_points(im0_not_resized)
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
