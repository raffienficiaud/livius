
import unittest
from tempfile import mkdtemp
import os
import shutil
import logging

from livius.video.processing.job import Job
from livius.video.processing.jobs.contrast_enhancement_boundaries import ContrastEnhancementBoundaries, \
    BoundariesConvolutionOnStableSegments
from livius.video.processing.jobs.segment_computation import SegmentComputationJob

from . import test_data_folder

FORMAT = '[%(asctime)-15s] %(message)s'
logging.basicConfig(format=FORMAT)

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


class JobTestsFixture(object):
    """Fixture for cleaning Job1/Job2 and the temporary directory"""

    def setUp(self):
        ContrastEnhancementBoundaries.parents = None
        BoundariesConvolutionOnStableSegments.parents = None

        self.tmpdir = '/media/renficiaud/linux-data/Code/EI/livius/tmp2'  #'/Users/raffi/Code/EI/livius/tmp2'  #mkdtemp()
        shutil.copy(os.path.join(test_data_folder, 'tests_video_contrast_enhancement_boundaries.json'),
                    self.tmpdir)
        shutil.copy(os.path.join(test_data_folder, 'tests_video_compute_segments.json'),
                    self.tmpdir)

        self.kwargs = {'segment_computation_tolerance': 0.02,
                       'json_prefix': os.path.join(self.tmpdir, 'tests_video'),
                       'segment_computation_min_length_in_seconds': 2}
        pass

    def tearDown(self):
        ContrastEnhancementBoundaries.parents = None
        BoundariesConvolutionOnStableSegments.parents = None

        #shutil.rmtree(self.tmpdir)


class ContrastEnhancementTests(JobTestsFixture, unittest.TestCase):

    def test_contrast_boundaries_setup(self):
        """Simple test that loads the known json file and asserts that
        the state does not need to be recomputed"""

        job_contrast = ContrastEnhancementBoundaries(**self.kwargs)
        job_contrast.load_state()

        self.assertTrue(job_contrast.is_up_to_date())

    def test_enhance_by_convolution_setup(self):
        """Simple test that sets up the convolution on the boundaries"""

        BoundariesConvolutionOnStableSegments.add_parent(ContrastEnhancementBoundaries)
        BoundariesConvolutionOnStableSegments.add_parent(SegmentComputationJob)

        job_contrast_conv = BoundariesConvolutionOnStableSegments(**self.kwargs)
        self.assertTrue(job_contrast_conv.contrast_enhancement_boundaries.is_up_to_date())
        self.assertFalse(job_contrast_conv.is_up_to_date())

    def test_enhance_by_convolution_process(self):
        """Simple processing step"""

        BoundariesConvolutionOnStableSegments.add_parent(ContrastEnhancementBoundaries)
        BoundariesConvolutionOnStableSegments.add_parent(SegmentComputationJob)

        job_contrast_conv = BoundariesConvolutionOnStableSegments(**self.kwargs)
        self.assertTrue(job_contrast_conv.contrast_enhancement_boundaries.is_up_to_date())
        self.assertFalse(job_contrast_conv.is_up_to_date())

        run_output = job_contrast_conv.process()
        self.assertIsNone(run_output)

        outputs = job_contrast_conv.get_outputs()
        self.assertIsNotNone(outputs)
        self.assertIsNotNone(outputs[0])
        self.assertIsNotNone(outputs[1])

        # checks that the internals are properly serializable
        job_contrast_conv.serialize_state()

        conv_out_min = []
        for i in xrange(max([k[1] for k in job_contrast_conv.compute_segments.get_outputs()])):
            conv_out_min += [outputs[0](i)]

        print conv_out_min[:50]

        import json
        with open(os.path.join(self.tmpdir, 'tests_video_conv.json'), 'w') as f:
            json.dump(conv_out_min, f)
