"""
Histogram Computation
=====================

This module provides a Job for the computation of the several histograms in
order to detect different changes in the scene (lightning, speaker motion, etc).

It also provides a Job for gathering all user input (slide location, speaker location).
"""

from ..job import Job
import cv2
import numpy as np
import functools

from ....util.tools import get_polygon_outer_bounding_box, crop_image_from_normalized_coordinates, \
                           sort_dictionary_by_integer_key
from ....util.functor import Functor
from .select_polygon import SelectPolygonJob


class HistogramsLABDiff(Job):

    """
    Job for computing histograms on polygons between two consecutive frames in specific areas of
    the plane.

    .. note:: The areas are specified by normalized coordinates


    **Parent inputs**

    The inputs of the parents are:

        * A list of tuples `(name, rectangle)` specifying the locations where the histogram should
          be computed.

            * `name` is indicating the name of the rectangle.
            * `rectangle` is given as `(x,y, width, height)`.

          .. note::
              If several rectangles exist for the same name, those are merged
              (which may be useful if the area is defined by several disconnected polygons).
        * A list of images specified by filename


    **Job outputs**

    The output of this Job is:

        * A function::

            frame_index, rectangle_name -> histogram

          that provides the histogram in the difference image in this particular rectangle.
    """

    name = 'histogram_imlabdiff'
    outputs_to_cache = ['histograms_labdiff']

    def __init__(self,
                 *args,
                 **kwargs):
        super(HistogramsLABDiff, self).__init__(*args, **kwargs)

    def load_state(self):
        """
        Sort the histograms by frame_index in order to be able to compare states.

        This is necessary because the json module can load and store dictionaries
        out of order.
        """
        state = super(HistogramsLABDiff, self).load_state()

        if state is None:
            return None

        histograms_labdiff = state['histograms_labdiff']

        for area in histograms_labdiff.keys():
            histograms_labdiff[area] = sort_dictionary_by_integer_key(histograms_labdiff[area])

        state['histograms_labdiff'] = histograms_labdiff
        return state

    def run(self, *args, **kwargs):
        assert(len(args) >= 2)

        self.rectangle_locations = args[0]

        image_list = args[1]

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

    def get_outputs(self):
        super(HistogramsLABDiff, self).get_outputs()
        if self.histograms_labdiff is None:
            raise RuntimeError('The points have not been selected yet')

        return Functor(self.histograms_labdiff, transform=functools.partial(np.array, dtype=np.float32))


class NumberOfVerticalStripes(Job):
    """Small Job that stores the number of vertical stripes used for speaker tracking."""

    name = 'number_of_vertical_stripes'
    # :
    outputs_to_cache = ['nb_vertical_stripes']

    def __init__(self, *args, **kwargs):
        super(NumberOfVerticalStripes, self).__init__(*args, **kwargs)
        assert('nb_vertical_stripes' in kwargs)

    def run(self, *args, **kwargs):
        pass

    def get_outputs(self):
        super(NumberOfVerticalStripes, self).get_outputs()

        return self.nb_vertical_stripes


# overriding some default behaviour with specific names
class SelectSlide(SelectPolygonJob):
    name = 'select_slides'
    window_title = 'Select the location of the Slides'


class SelectSpeaker(SelectPolygonJob):
    name = 'select_speaker'
    window_title = 'Select the location of the Speaker'


class GatherSelections(Job):

    """
    Job for combining the polygon selections for the slides & speaker location.


    **Job outputs**

    The output of this Job is:

        * A list of tuples `(name, rect)` where each tuples contains
            * The name of the area
            * A normalized rectangle `[x,y,width,height]` that specifies the area.
    """

    name = 'gather_selections'
    parents = [SelectSlide, SelectSpeaker, NumberOfVerticalStripes]
    outputs_to_cache = ['rectangle_locations']

    def run(self, *args, **kwargs):
        self.rectangle_locations = []

        # slide location gives the position where to look for the illumination
        # changes detection
        slide_loc = args[0]
        slide_rec = get_polygon_outer_bounding_box(slide_loc)
        x, y, width, height = slide_rec

        first_light_change_area = [0, y, x, height]
        second_light_change_area = [x + width, y, 1 - (x + width), height]

        # @note(Stephan): Unicode names in order to compare to the json file
        self.rectangle_locations += [u'slides', first_light_change_area], \
                                    [u'slides', second_light_change_area]

        # speaker location is divided into vertical stripes on the full horizontal
        # extent
        speaker_loc = args[1]
        speaker_rec = get_polygon_outer_bounding_box(speaker_loc)
        _, y, _, height = speaker_rec
        nb_vertical_stripes = args[2]

        width_stripes = 1.0 / nb_vertical_stripes
        for i in range(nb_vertical_stripes - 1):
            x_start = width_stripes * i
            rect_stripe = [x_start, y, width_stripes, height]
            self.rectangle_locations += [u'speaker_%.2d' % i,
                                         rect_stripe],

        # final stripe adjusted a bit to avoid getting out the image plane
        rect_stripe = [1 - width_stripes, y, width_stripes, height]
        self.rectangle_locations += [u'speaker_%.2d' % (nb_vertical_stripes - 1),
                                     rect_stripe],

    def get_outputs(self):
        super(GatherSelections, self).get_outputs()

        if self.rectangle_locations is None:
            raise RuntimeError('The Areas we want to compute the histograms on have not been computed yet.')

        return self.rectangle_locations
