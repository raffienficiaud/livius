"""
This file provides the Job interface for the computation of the several histograms in
order to detect different changes in the scene (lightning, speaker motion, etc).
"""

from ..job import Job
import os
import cv2
import numpy as np
import functools

from ....util.tools import get_polygon_outer_bounding_box, crop_image_from_normalized_coordinates, \
                           sort_dictionary_by_integer_key
from ....util.functor import Functor
from .select_polygon import SelectPolygonJob


class HistogramsLABDiff(Job):

    """
    Computes histogram on polygons between two consecutive frames in specific areas of the plane.
    (normalized coordinates)

    Inputs of the parents:
    - a list of `(name, rectangle)` specifying the locations where the histogram should be computed,
      The `name` indicating the name of the rectangle.
      The `rectangle` is given as (x,y, width, height).
      If several rectangles exist for the same name, those are merged
      (which may be useful if the area is defined by several disconnected polygons).
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
        super(HistogramsLABDiff, self).__init__(*args, **kwargs)

    def load_state(self):
        """Sort the histograms by frame_index in order to be able to compare states."""
        state = super(HistogramsLABDiff, self).load_state()

        if state is None:
            return None

        histograms_labdiff = state['histograms_labdiff']

        for area in histograms_labdiff.keys():
            histograms_labdiff[area] = sort_dictionary_by_integer_key(histograms_labdiff[area])

        state['histograms_labdiff'] = histograms_labdiff
        return state

    def is_up_to_date(self):
        """
        Return False if no correlation has been computed (or can be restored from the json dump),
        default behaviour otherwise.
        """
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

        rectangle_names = zip(*self.rectangle_locations)[0]
        unique_rectangle_names = list(set(rectangle_names))

        for name in unique_rectangle_names:
            element = self.histograms_labdiff.get(name, {})
            self.histograms_labdiff[name] = element

        # perform the computation
        im_index_tm1 = cv2.imread(image_list[0])
        imlab_index_tm1 = cv2.cvtColor(im_index_tm1, cv2.COLOR_BGR2LAB)

        for index, filename in enumerate(image_list[1:], 1):
            im_index_t = cv2.imread(filename)
            imlab_index_t = cv2.cvtColor(im_index_t, cv2.COLOR_BGR2LAB)

            # color diff
            im_diff = (imlab_index_t - imlab_index_tm1) ** 2
            im_diff_lab = np.sqrt(np.sum(im_diff, axis=2))

            # Compute histogram for every area
            for name, rect in self.rectangle_locations:
                cropped = crop_image_from_normalized_coordinates(im_diff_lab, rect)
                histogram = cv2.calcHist([cropped.astype(np.uint8)], [0], None, [256], [0, 256])

                # Merge histograms if necessary
                histogram_to_merge = self.histograms_labdiff[name].get(index, None)
                if histogram_to_merge is not None:
                    histogram += histogram_to_merge

                self.histograms_labdiff[name][index] = histogram

            # @note(Stephan):
            # The histograms are stored as a python list in order to serialize them via JSON.
            for name in unique_rectangle_names:
                histogram_np_array = self.histograms_labdiff[name][index]
                self.histograms_labdiff[name][index] = histogram_np_array.tolist()

        # save the state (commit to json)
        self.serialize_state()

    def get_outputs(self):
        super(HistogramsLABDiff, self).get_outputs()
        if self.histograms_labdiff is None:
            raise RuntimeError('The points have not been selected yet')

        return Functor(self.histograms_labdiff, transform=functools.partial(np.array, dtype=np.float32))


# overriding some default behaviour with specific names
class SelectSlide(SelectPolygonJob):
    name = 'select_slides'
    window_title = 'Select the location of the Slides'


class SelectSpeaker(SelectPolygonJob):
    name = 'select_speaker'
    window_title = 'Select the location of the Speaker'


class GatherSelections(Job):

    """
    This job combines the polygon selections for the slides & speaker location.

    The output is:
        A list of tuples where each tuples contains
        - The name of the area
        - and a normalized rectangle [x,y,width,height] that specifies the area.
    """

    name = 'gather_selections'
    parents = [SelectSlide, SelectSpeaker]
    attributes_to_serialize = ['nb_vertical_stripes']

    def __init__(self, *args, **kwargs):
        super(GatherSelections, self).__init__(*args, **kwargs)

        assert('nb_vertical_stripes' in kwargs)

    def run(self, *args, **kwargs):
        pass

    def get_outputs(self):

        list_polygons = []

        # slide location gives the position where to look for the illumination
        # changes detection
        slide_loc = self.select_slides.get_outputs()
        slide_rec = get_polygon_outer_bounding_box(slide_loc)
        x, y, width, height = slide_rec

        first_light_change_area = [0, y, x, height]
        second_light_change_area = [x + width, y, 1 - (x + width), height]

        # @note(Stephan): Unicode names in order to compare to the json file
        list_polygons += [u'slides', first_light_change_area], \
                         [u'slides', second_light_change_area]

        # speaker location is divided into vertical stripes on the full horizontal
        # extent
        speaker_loc = self.select_speaker.get_outputs()
        speaker_rec = get_polygon_outer_bounding_box(speaker_loc)
        _, y, _, height = speaker_rec

        width_stripes = 1.0 / self.nb_vertical_stripes
        for i in range(self.nb_vertical_stripes - 1):
            x_start = width_stripes * i
            rect_stripe = [x_start, y, width_stripes, height]
            list_polygons += [u'speaker_%.2d' % i,
                              rect_stripe],

        # final stripe adjusted a bit to avoid getting out the image plane
        rect_stripe = [1 - width_stripes, y, width_stripes, height]
        list_polygons += [u'speaker_%.2d' % (self.nb_vertical_stripes - 1),
                          rect_stripe],

        return list_polygons

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
