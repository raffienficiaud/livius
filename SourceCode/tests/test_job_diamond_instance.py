'''
Tests that jobs with the same name in the workflow are not duplicated
'''

from .test_job import JobTestsFixture
import unittest
import os

# logging facility
import logging
FORMAT = '[%(asctime)-15s] %(message)s'
logging.basicConfig(format=FORMAT)

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

from ..video.processing.job import Job


class J1(Job):
    name = 'j1'
    def run(self, *args, **kwargs):
        pass

class J2(Job):
    name = 'j2'
    parents = [J1]
    def run(self, *args, **kwargs):
        pass

class J3(Job):
    name = 'j3'
    parents = [J1]
    def run(self, *args, **kwargs):
        pass

class J4(Job):
    name = 'j4'
    parents = [J2, J3]
    def run(self, *args, **kwargs):
        pass

class J5(Job):
    name = 'j5'
    parents = [J4, J1]
    def run(self, *args, **kwargs):
        pass


class JobDiamondTest(JobTestsFixture, unittest.TestCase):

    def test_parents_same_instance_on_graph(self):
        job_final = J5(json_prefix=os.path.join(self.tmpdir, 'test_meta'))

        print job_final._parent_names

        self.assertIs(job_final.j1, job_final.j4.j3.j1)
