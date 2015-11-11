"""Tests the correct caching of parameters"""


import unittest
from tempfile import mkdtemp
import os
import shutil
import tempfile

# logging facility
import logging
FORMAT = '[%(asctime)-15s] %(message)s'
logging.basicConfig(format=FORMAT)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

from ..video.processing.job import Job


class JDummy(Job):
    name = 'jdummy'
    attributes_to_serialize = ['some_parameter']
    output_to_cache = ['some_output']

    def __init__(self, *args, **kwargs):
        super(JDummy, self).__init__(*args, **kwargs)

    def run(self, *args, **kwargs):
        self.some_output = self.some_parameter
        pass

    def get_outputs(self):
        super(JDummy, self).get_outputs()
        return self.some_output


class JobCacheTest(unittest.TestCase):

    def setUp(self):
        self.temporary_folder = tempfile.mkdtemp()
        if not os.path.exists(self.temporary_folder):
            os.makedirs(self.temporary_folder)

    def tearDown(self):
        shutil.rmtree(self.temporary_folder)

    def get_job(self, **kwargs):
        return JDummy(json_prefix=self.temporary_folder, **kwargs)

    def test_cache_array_int(self):
        """Tests the ability to cache dictionaries containing strings"""

        some_parameter = [1, 1, 1, 1, 3]
        job = self.get_job(some_parameter=some_parameter)
        job.process()

        self.assertTrue(job.is_up_to_date())

        out = job.get_outputs()
        self.assertEqual(some_parameter, out)

        job2 = self.get_job(some_parameter=some_parameter)
        self.assertTrue(job2.is_up_to_date())


    def test_cache_dict_string(self):
        """Tests the ability to cache dictionaries containing strings"""

        some_parameter = {"param1": "value1", "param2": "value2"}
        job = self.get_job(some_parameter=some_parameter)
        job.process()

        self.assertTrue(job.is_up_to_date())

        out = job.get_outputs()
        self.assertEqual(some_parameter, out)

        job2 = self.get_job(some_parameter=some_parameter)
        self.assertTrue(job2.is_up_to_date())
