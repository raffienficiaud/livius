"""
Select Polygon
==============

This module defines a Job for selecting a specific area from a video file name.
"""

import os
import cv2
from random import randint

from ..job import Job
from ....util.user_interaction import get_polygon_from_user


class SelectPolygonJob(Job):

    """
    This Job shows one frame of the input video (or extracted thumbnail image) to the user and asks
    for a polygon defining the location of the slides


    **Job parameters**

    :param video_filename: The name of the video file to process


    **Job outputs**

    * The selected points as a list of normalized coordinates `[x, y]`
    """

    name = "select_polygon"
    #:
    attributes_to_serialize = ['video_filename']
    #:
    outputs_to_cache = ['points']
    #: Specifies the window title that is shown to the user when asked to perform the area selection.
    window_title = ''

    def __init__(self,
                 *args,
                 **kwargs):
        """
        Expected parameters in kwargs:
        :param video_filename: The name of the video file to process
        """
        super(SelectPolygonJob, self).__init__(*args, **kwargs)

        video_filename = kwargs.get('video_filename', None)
        if video_filename is None:
            raise RuntimeError("The video file name cannot be empty")
        if not os.path.exists(video_filename):
            raise RuntimeError("The video file %s does not exist" % os.path.abspath(video_filename))

        # this is necessary because the json files are stored in unicode, and the
        # comparison of the list of files should work (unicode path operations
        # is unicode)
        self.video_filename = os.path.abspath(unicode(video_filename))

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

        self.points = get_polygon_from_user(im_for_selection, 4, self.window_title)

        # @note(Stephan):
        # Since Json stores tuples as list, we go from tuples to lists here. Then we can compare the
        # self.points attribute directly to the json file (as in are_states_equal)
        self.points = [[float(i) / width, float(j) / height] for (i, j) in self.points]

    def get_outputs(self):
        super(SelectPolygonJob, self).get_outputs()
        if self.points is None:
            raise RuntimeError('The points have not been selected yet')
        return self.points
