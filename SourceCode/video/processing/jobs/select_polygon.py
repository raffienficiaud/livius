"""
Select Polygon
==============

This module defines a unique jobs for selecting polygons on images and/or videos.

.. autosummary::

  SelectPolygonJob
  SelectSlide
  SelectSpeaker

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

    .. rubric:: Runtime parameter

    See :py:func:`__init__`.

    .. rubric:: Workflow input

    * either nothing, in which case the video file is used
    * or a list of image file names, in which case a randomly picked image is
      shown for input.

    .. rubric:: Workflow output

    The selected points as a list of normalized coordinates `[x, y]`

    """

    #: name of the job in the workflow
    name = "select_polygon"

    #: Cached inputs:
    #:
    #: * ``video_filename`` the name of the input video without the path name
    attributes_to_serialize = ['video_filename']

    #: Cached outputs:
    #:
    #: * ``points``: location of the points of the selected polygon, in normalized coordinates
    outputs_to_cache = ['points']

    #: Specifies the window title that is shown to the user when asked to perform the area selection
    window_title = ''


    def __init__(self,
                 *args,
                 **kwargs):
        """
        Expected parameters in kwargs:

        :param video_filename: The name of the video file to process, without the folder
        :param video_location: the directory where the video file is (not cached for relation purposes).
          This parameter is mandatory
        """
        super(SelectPolygonJob, self).__init__(*args, **kwargs)

        if self.video_filename is None:
            raise RuntimeError("The video file name cannot be empty")

        assert('video_location' in kwargs)

        if not os.path.exists(os.path.join(self.video_location, self.video_filename)):
            raise RuntimeError("The video file %s does not exist" % os.path.abspath(self.video_filename))

        # this `unicode` necessary because the json files are stored in unicode, and the
        # comparison of the list of files should work (unicode path operations
        # is unicode)
        self.video_filename = unicode(self.video_filename)

    def run(self, *args, **kwargs):

        if self.is_up_to_date():
            return True

        if not args:
            cap = cv2.VideoCapture(os.path.join(self.video_location, self.video_filename))
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


class SelectSlide(SelectPolygonJob):
    """User input for slide location.

    Refines the behaviour of :py:class:`SelectPolygonJob`.
    """

    #: name of the job in the workflow
    name = 'select_slides'

    window_title = 'Select the location of the Slides'


class SelectSpeaker(SelectPolygonJob):
    """User input for speaker location.

    Refines the behaviour of :py:class:`SelectPolygonJob`.
    """

    #: name of the job in the workflow
    name = 'select_speaker'

    window_title = 'Select the location of the Speaker'



def selection_mogrifier(folder, outfolder):
    """Utility function transforming existing selections ready for
    :py:class:`SelectSlide <SourceCode.video.processing.jobs.select_polygon.SelectSlide>` and
    :py:class:`SelectSpeaker <SourceCode.video.processing.jobs.select_polygon.SelectSpeaker>`.

    This is a simple function reading 2 files in a JSON format in an old (unsupported) format and
    writing them to the output folder in the new format.

    It is possible to run directly this function by providing the location of the folder to parse, in the following way::

      python -m SourceCode.video.processing.jobs.select_polygon some_folder_init output_folder_root

    """
    import json

    list_videos = os.listdir(folder)

    for current_video in list_videos:

        print 'processing video', current_video

        full_path = os.path.abspath(os.path.join(folder, current_video))
        out_path = os.path.abspath(os.path.join(outfolder, current_video))

        if not os.path.exists(out_path):
            os.makedirs(out_path)

        video_filename = current_video

        # speaker
        dout = {}
        selection1 = os.path.join(full_path, 'processing_' + video_filename + '__select_speaker.json')
        with open(selection1) as f:
            d = json.load(f)
            dout['points'] = d['points']
            dout['video_filename'] = os.path.basename(d['video_filename'])

        out_file = os.path.join(out_path, video_filename + '_select_speaker.json')
        with open(out_file, 'w') as f:
            json.dump(dout, f, indent=4)

        # slides
        dout = {}
        selection2 = os.path.join(full_path, 'processing_' + video_filename + '__select_slides.json')
        with open(selection2) as f:
            d = json.load(f)
            dout['points'] = d['points']
            dout['video_filename'] = os.path.basename(d['video_filename'])

        out_file = os.path.join(out_path, video_filename + '_select_slides.json')
        with open(out_file, 'w') as f:
            json.dump(dout, f, indent=4)

if __name__ == '__main__':
    import sys
    print sys.argv
    selection_mogrifier(sys.argv[1], sys.argv[2])
