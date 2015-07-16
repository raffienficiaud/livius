"""
This file provides the Job interface for the computation of the several histograms in
order to detect different changes in the scene (lightning, speaker motion, etc).
"""

from ..job import Job
import os
import json
import cv2
import numpy as np

from ....util.tools import get_polygon_outer_bounding_box, crop_image_from_normalized_coordinates


class HistogramsLABDiff(Job):
    """
    Computes histogram on polygons between two consecutive frames in specific areas of the plane (normalized coordinates).

    Expect parameters from parents, in this order:

    - a list of `(name, polygon)` specifying the locations where the histogram should be computed, the `name` indicating the
      name of the rectangle. The `polygon` is transformed into its bounding box using :py:`get_polygon_outer_bounding_box`.
      If several polygons exist for the same name, those are merged (which may be useful if the area is defined by
      several disconnected polygons).
    - a list of images

    The output is:
    - a function of frame index and rectangle name (two arguments) that provides the histogram
      in the difference image in this particular rectangle

    The state of this function is saved on the json file.

    """

    name = 'histogram_imlabdiff'
    attributes_to_serialize = ['rectangle_locations',
                               'number_of_files',
                               'histograms_labdiff'
                               ]

    def __init__(self,
                 *args,
                 **kwargs):
        """
        :param :
        """
        super(HistogramsLABDiff, self).__init__(*args, **kwargs)

        self._get_previous_state()

        # read back the output files if any
        pass

    def _get_previous_state(self):
        if not os.path.exists(self.json_filename):
            return

        with open(self.json_filename) as f:
            d = json.load(f)

            # maybe take a subset of attributes
            for k in self.attributes_to_serialize:
                if k in d:
                    setattr(self, k, d[k])

    def is_up_to_date(self):
        """Returns False if no correlation has been computed (or can be restored from
        the json dump), default behaviour otherwise"""
        if not self.histograms_labdiff \
           or not self.number_of_files:
            return False

        return super(HistogramsLABDiff, self).is_up_to_date()

    def run(self, *args, **kwargs):
        assert(len(args) >= 2)

        self.rectangle_locations = args[0]

        image_list = args[1]
        assert(len(image_list) == self.number_of_files or self.number_of_files is None)
        self.number_of_files = len(image_list)

        # init
        self.histograms_labdiff = {}

        # perform the computation
        im_index_tm1 = cv2.imread(image_list[0])
        imlab_index_tm1 = cv2.cvtColor(im_index_tm1, cv2.COLOR_BGR2LAB)

        for index, filename in enumerate(image_list[1:], 1):
            im_index_t = cv2.imread(filename)
            imlab_index_t = cv2.cvtColor(im_index_t, cv2.COLOR_BGR2LAB)

            # color diff
            im_diff = (imlab_index_t - imlab_index_tm1) ** 2
            im_diff_lab = np.sqrt(np.sum(im_diff, axis=2))

            for name, polygon in self.rectangle_location:

                rect = get_polygon_outer_bounding_box(polygon)
                cropped_image = crop_image_from_normalized_coordinates(im_diff_lab, rect)
                self.histograms_labdiff[name][index] = cv2.calcHist([cropped_image.astype(np.uint8)], [0], None, [256], [0, 256])

            pass

        # save the state (commit to json)
        self.serialize_state()

    def get_outputs(self):
        super(HistogramsLABDiff, self).get_outputs()
        if self.histograms_labdiff is None:
            raise RuntimeError('The points have not been selected yet')

        class FeatureTime(object):

            def __init__(self, histograms_per_frame_rect):
                self.histograms_per_frame_rect = histograms_per_frame_rect
                return

            def __call__(self, frame_index, area_name):
                return self.histograms_per_frame_rect[frame_index][area_name]

        return FeatureTime(self.histograms_labdiff)


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

    from .select_polygon import SelectPolygonJob

    # overriding some default behaviour with specific names

    class SelectSlide(SelectPolygonJob):
        name = 'select_slides'

    class SelectSpeaker(SelectPolygonJob):
        name = 'select_speaker'

    class GatherSelections(Job):
        name = 'gather_selections'
        parents = [SelectSlide, SelectSpeaker]
        nb_vertical_stripes = 10
        attributes_to_serialize = ['nb_vertical_stripes']

        def run(self, *args, **kwargs):
            pass

        def get_outputs(self):

            list_polygons = []

            # slide location gives the position where to look for the illumination
            # changes detection
            slide_loc = self.select_slides.get_outputs()
            first_light_change_area = [0, slide_loc[1], slide_loc[0], slide_loc[3]]
            second_light_change_area = [slide_loc[0] + slide_loc[2], slide_loc[1],
                                        1 - (slide_loc[0] + slide_loc[2]), slide_loc[3]]
            list_polygons += ('slides', first_light_change_area), \
                             ('slides', second_light_change_area)

            # speaker location is divided into vertical stripes on the full horizontal
            # extent
            speaker_loc = self.select_speaker.get_outputs()

            width_stripes = 1.0 / self.nb_vertical_stripes
            for i in range(self.nb_vertical_stripes - 1):
                x_start = width_stripes * i
                rect_stripe = [x_start, speaker_loc[1], width_stripes, speaker_loc[3]]
                list_polygons += ('speaker_%.2d' % i,
                                  rect_stripe)

            # final stripe adjusted a bit to avoid getting out the image plane
            rect_stripe = [1 - width_stripes, speaker_loc[1], width_stripes, speaker_loc[3]]
            list_polygons += ('speaker_%.2d' % self.nb_vertical_stripes,
                              rect_stripe)

            return list_polygons

    from .ffmpeg_to_thumbnails import FFMpegThumbnailsJob
    HistogramsLABDiff.add_parent(GatherSelections)
    HistogramsLABDiff.add_parent(FFMpegThumbnailsJob)

    # import ipdb
    d = {'video_filename': current_video,
         'thumbnails_location': os.path.join(proc_folder, 'processing_video_7_thumbnails'),
         'json_prefix': os.path.join(proc_folder, 'processing_video_7_')}

    job_instance = HistogramsLABDiff(**d)
    job_instance.process()

    # should not pop out a new window because same params
    job_instance2 = HistogramsLABDiff(**d)
    job_instance2.process()
