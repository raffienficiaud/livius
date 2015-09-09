"""
Segment Computation
===================

This module defines a Job for extracting the stable segments of the video
from the histogram correlations.
"""

from ..job import Job


class SegmentComputationJob(Job):

    """
    Job for splitting the video into stable segments.

    .. note::
        We assume that we can apply a constant contrast enhancement for the slides in those
        stable parts of the video.


    **Job parameters**
    Parameters of the Job (expected to be passed when constructing a workflow instance):

    :param segment_computation_tolerance:
        How much deviation from correlation 1.0 do we allow

    :param segment_computation_min_length_in_seconds:
        The minimum length of a stable segment. If segments are shorter, we count it as not
        being stable.


    **Parent inputs**

    The inputs of the parents are:
        * A function::

            frame_index -> correlation

          (:py:class:`.histogram_correlations.HistogramCorrelationJob`)

        * The number of files


    **Job outputs**

    The output of this Job is:
        * A list of segments, each specified by `[t_start, t_end]`.
    """

    name = "compute_segments"
    #:
    attributes_to_serialize = ['segment_computation_tolerance',
                               'segment_computation_min_length_in_seconds']
    #:
    outputs_to_cache = ['segments']

    def __init__(self,
                 *args,
                 **kwargs):

        super(SegmentComputationJob, self).__init__(*args,
                                                    **kwargs)

        assert('segment_computation_tolerance' in kwargs)
        assert('segment_computation_min_length_in_seconds' in kwargs)

    def run(self, *args, **kwargs):
        """
        Segment the video using the histogram correlations.

        If there is a spike with a correlation of less than (1 - tolerance), then
        we assume that we need to adapt the histogram bounds and thus start a
        new segment.

        If there is a region with many small spikes we assume that we cannot apply
        any contrast enhancement / color correction (or apply a conservative default one).
        """
        # First parent is HistogramCorrelationJob.
        get_histogram_correlation = args[0]

        # Second Parent is the Number of Files
        number_of_files = args[1]

        self.segments = []

        t_segment_start = 0.0
        lower_bounds = 1.0 - self.segment_computation_tolerance

        # @todo(Stephan):
        # This information should probably be passed together with the histogram differences
        seconds_per_correlation_entry = 1

        # @note(Stephan): The first correlation can be computed at frame_index 2, so we start from there.
        i = 2
        end = number_of_files

        t = i * seconds_per_correlation_entry

        while i < end:

            # As long as we stay over the boundary, we count it towards the same segment
            while (i < end) and (get_histogram_correlation(i) >= lower_bounds):
                i += 1
                t += seconds_per_correlation_entry

            # Append segment if it is big enough
            if (t - t_segment_start) >= self.segment_computation_min_length_in_seconds:
                self.segments.append([t_segment_start, t])

            # Skip the elements below the boundary
            while (i < end) and (get_histogram_correlation(i) < lower_bounds):
                i += 1
                t += seconds_per_correlation_entry

            # The new segment starts as soon as we are over the boundary again
            t_segment_start = t

    def get_outputs(self):
        super(SegmentComputationJob, self).get_outputs()
        if self.segments is None or len(self.segments) == 0:
            raise RuntimeError('The segments have not been computed yet')
        return self.segments
