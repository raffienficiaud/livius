"""
Histogram Correlations
======================

This module provides the Job for computation of the Histogram correlations that are used to
determine the stable segments of the video.
"""

from ..job import Job

import cv2

from ....util.functor import Functor
from ....util.tools import sort_dictionary_by_integer_key


class HistogramCorrelationJob(Job):

    """
    Job for the computation of the Histogram correlations.


    **Parent inputs**

    The inputs of the parents are:
        * A function::

            (frame_index, area_name) -> Histogram

          (:py:class:`.histogram_computation.HistogramsLABDiff`)

        * The number of thumbnails
        * The number of vertical stripes

    **Job outputs**

    The output of this Job is:
        * A function::

            frame_index -> HistogramCorrelation

    """

    name = 'histogram_correlation'
    #:
    attributes_to_serialize = []
    #:
    outputs_to_cache = ['histogram_correlations']

    def __init__(self, *args, **kwargs):
        super(HistogramCorrelationJob, self).__init__(*args, **kwargs)

    def load_state(self):
        state = super(HistogramCorrelationJob, self).load_state()

        if state is None:
            return None

        correlations = state['histogram_correlations']
        correlations = sort_dictionary_by_integer_key(correlations)

        state['histogram_correlations'] = correlations

        return state

    def run(self, *args, **kwargs):

        # The first parent is the HistogramComputation
        get_histogram = args[0]

        # Second parent is the NumberOfFiles
        number_of_files = args[1]

        # init
        self.histogram_correlations = {}

        # @todo(Stephan): We need the number of vertical stripes
        nb_vertical_stripes = 10
        speaker_area_names = ['speaker_%.2d' % i for i in range(nb_vertical_stripes)]

        def get_speaker_histogram_plane(frame_index):
            return map(lambda area_name: get_histogram(area_name, frame_index), speaker_area_names)

        previous_slide_histogram = get_histogram('slides', 1)
        previous_speaker_histogram_plane = get_speaker_histogram_plane(1)

        for frame_index in range(2, number_of_files):

            slide_histogram = get_histogram('slides', frame_index)
            speaker_histogram_plane = get_speaker_histogram_plane(frame_index)

            if previous_slide_histogram is not None:
                self.histogram_correlations[frame_index] = \
                    cv2.compareHist(slide_histogram, previous_slide_histogram, cv2.cv.CV_COMP_CORREL)

            if previous_speaker_histogram_plane is not None:
                # @todo(Stephan): Compute the energy here?
                # Maybe create a new Job and a common subclass (cf. GatherSelections)
                pass

            previous_slide_histogram = slide_histogram
            previous_speaker_histogram_plane = speaker_histogram_plane

    def get_outputs(self):
        super(HistogramCorrelationJob, self).get_outputs()

        if self.histogram_correlations is None:
            raise RuntimeError('The Correlations between the histograms have not been computed yet.')

        return Functor(self.histogram_correlations)
