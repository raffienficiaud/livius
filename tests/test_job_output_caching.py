"""
Test that the outputs of nodes that are up to date are loaded from the JSON file and not recomputed.
"""

import unittest
import os

from livius.video.processing.job import Job

from .test_job import JobTestsFixture


class J1(Job):
    name = 'j1'
    attributes_to_serialize = ['j1_attr1', 'j1_attr2']
    outputs_to_cache = ['j1_out']

    def run(self, *args, **kwargs):
        pass

    def get_outputs(self):
        return getattr(self, 'j1_out')


class J2(Job):
    name = 'j2'
    attributes_to_serialize = ['j2_attr1']
    outputs_to_cache = ['j2_out']
    parents = [J1]

    def run(self, *args, **kwargs):
        pass

    def get_outputs(self):
        return getattr(self, 'j2_out')


class J3(Job):
    name = 'j3'
    attributes_to_serialize = ['j3_attr1']
    outputs_to_cache = ['has_run']

    def __init__(self, *args, **kwargs):
        super(J3, self).__init__(*args, **kwargs)
        self.has_run = False

    def run(self, *args, **kwargs):
        self.has_run = True

    def get_outputs(self):
        super(J3, self).get_outputs()
        pass


class TestOutputCaching(JobTestsFixture, unittest.TestCase):
    def setUp(self):
        super(TestOutputCaching, self).setUp()

        self.json_prefix = os.path.join(self.tmpdir, 'test_output_cache')

        args = {'json_prefix': self.json_prefix,
                'j1_attr1': 11,
                'j1_attr2': 12,
                'j2_attr1': 21,
                'j1_out': 0.1,
                'j2_out': 0.2}

        self.j2 = J2(**args)

    def test_output_cache(self):
        self.j2.process()

        self.assertEquals(self.j2.get_outputs(), 0.2)
        self.assertTrue(self.j2.is_output_cached())
        self.assertTrue(self.j2.j1.is_output_cached())

        # Delete the output of j2 (e.g when running the workflow again)
        self.j2.j2_out = None

        # Output is not cached anymore
        self.assertFalse(self.j2.is_output_cached())
        # The state of the Job should still be the same!
        self.assertTrue(self.j2.is_up_to_date())

        # Load the state back from the JSON file
        self.j2.cache_output()
        self.assertTrue(self.j2.is_output_cached())
        self.assertEquals(self.j2.get_outputs(), 0.2)

    def test_has_run(self):
        args = {'json_prefix': os.path.join(self.tmpdir, 'test_has_run'),
                'j3_attr1': 31}
        j3 = J3(**args)
        self.assertFalse(j3.has_run)

        j3.process()
        self.assertTrue(j3.has_run)

        j3.has_run = False

        # Should not be run again, because has_run is not a parameter, but a cached output.
        j3.process()
        self.assertFalse(j3.has_run)
