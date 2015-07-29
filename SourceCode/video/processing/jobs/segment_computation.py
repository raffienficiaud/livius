"""Defines a job for extracting the segments from the histogram correlations."""

import os
import json

from ..job import Job


class SegmentComputationJob(Job):

    """
    Splits the video into stable segments.

    We assume that we can apply a constant contrast enhancement for the slides in those Segments.

    Parameters of the Job (expected to be passed when constructing a workflow instance):
        - segment_computation_tolerance:
            How much deviation from correlation 1.0 do we allow

        - segment_computation_min_length_in_seconds:
            The minimum length of a stable segment. If segments are shorter, we count it as not
            being stable.


    Inputs of the parents:
        - a function :: frame_index -> correlation
        - The number of files


    The output is:
        A list of segments, each specified by [t_start, t_end].
    """

    name = "compute_segments"
    attributes_to_serialize = ['segment_computation_tolerance',
                               'segment_computation_min_length_in_seconds',
                               'segments']

    def __init__(self,
                 *args,
                 **kwargs):

        super(SegmentComputationJob, self).__init__(*args,
                                                    **kwargs)

        assert('segment_computation_tolerance' in kwargs)
        assert('segment_computation_min_length_in_seconds' in kwargs)

        self._get_previously_computed_segments()

    def _get_previously_computed_segments(self):
        if not os.path.exists(self.json_filename):
            return None

        with open(self.json_filename) as f:
            d = json.load(f)

            if 'segments' in d:
                setattr(self, 'segments', d['segments'])

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

        self.serialize_state()

    def get_outputs(self):
        super(SegmentComputationJob, self).get_outputs()
        if self.segments is None or len(self.segments) == 0:
            raise RuntimeError('The segments have not been computed yet')
        return self.segments


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

    from .histogram_computation import HistogramsLABDiff, GatherSelections
    from .ffmpeg_to_thumbnails import FFMpegThumbnailsJob
    from .histogram_correlations import HistogramCorrelationJob

    HistogramsLABDiff.add_parent(GatherSelections)
    HistogramsLABDiff.add_parent(FFMpegThumbnailsJob)

    HistogramCorrelationJob.add_parent(HistogramsLABDiff)

    SegmentComputationJob.add_parent(HistogramCorrelationJob)

    # import ipdb
    d = {'video_filename': current_video,
         'thumbnails_location': os.path.join(proc_folder, 'processing_video_7_thumbnails'),
         'json_prefix': os.path.join(proc_folder, 'processing_video_7_'),
         'segment_computation_tolerance': 0.05,
         'segment_computation_min_length_in_seconds': 2}

    job_instance = SegmentComputationJob(**d)
    job_instance.process()
