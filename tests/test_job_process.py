'''
Test the proper processing of the Job and their parents
'''

import unittest
import os

from livius.video.processing.job import Job

from .test_job import JobTestsFixture, Job1, Job2


class JobProcessTests(JobTestsFixture, unittest.TestCase):
    """Tests the proper processing of a DAG of Jobs"""

    def setUp(self):
        # this is a local class
        class Job1_der(Job1):
            name = 'job1_der'

            def run(self, *args, **kwargs):
                self.args = args
                self.kwargs = kwargs
                pass

            def get_outputs(self):
                return [1, 2, 3, 4, 5]

        super(JobProcessTests, self).setUp()
        Job1.attributes_to_serialize.append('job1attr1')
        Job1.attributes_to_serialize.append('job1attr2')
        Job2.attributes_to_serialize.append('job2attr1')
        Job2.add_parent(Job1_der)

    # def tearDown(self):
    #    super(JobProcessTests, self).tearDown()

    def test_process(self):

        job_final = Job2(json_prefix=os.path.join(self.tmpdir, 'test_process'))
        job_final.process()

        job1_der_instance = job_final.get_parent_by_name('job1_der')
        self.assertFalse(job1_der_instance.args)
        self.assertFalse(job1_der_instance.kwargs)

        self.assertTrue(job_final.args)
        self.assertFalse(job_final.kwargs)

        pass

    def test_process_arg_flow(self):
        class Job2_der(Job2):
            name = 'job2_der'

            def is_up_to_date(self):
                if hasattr(self, 'result'):
                    return True
                return False

            def run(self, *args, **kwargs):
                self.args = args
                self.kwargs = kwargs
                self.result = [i ** 2 for i in args[0]]
                pass

            def get_outputs(self):
                return self.result

        job_final = Job2_der(json_prefix=os.path.join(self.tmpdir, 'test_process'))
        self.assertFalse(job_final.is_up_to_date())
        job_final.process()
        self.assertEqual(job_final.get_outputs(), [i ** 2 for i in [1, 2, 3, 4, 5]])
        self.assertTrue(job_final.is_up_to_date())

        # should not process it again
        job_final.result = []
        job_final.process()
        self.assertEqual(job_final.get_outputs(), [])

    def test_process_arg_flow_diamond(self):

        class ResultJob(Job):
            def is_up_to_date(self):
                if hasattr(self, 'result'):
                    return True
                return False

            def get_outputs(self):
                return self.result

        class Job2_left(ResultJob, Job):
            name = 'Job2_left'
            parents = [Job2]

            def run(self, *args, **kwargs):
                self.args = args
                self.kwargs = kwargs
                self.result = [i ** 2 for i in args[0]]
                pass

        class Job2_right(ResultJob, Job):
            name = 'Job2_right'
            parents = [Job2]

            def run(self, *args, **kwargs):
                self.args = args
                self.kwargs = kwargs
                self.result = [i for i in args[0]]
                pass

        class JobFinal(ResultJob, Job):
            name = 'JobFinal'
            parents = [Job2_left, Job2_right]

            def run(self, *args, **kwargs):
                self.args = args
                self.kwargs = kwargs
                self.result = [a * b for (a, b) in zip(*args)]
                pass

        job_final = JobFinal(json_prefix=os.path.join(self.tmpdir, 'test_diamond'))
        self.assertFalse(job_final.is_up_to_date())
        job_final.process()
        self.assertTrue(job_final.is_up_to_date())

        self.assertEqual(job_final.get_outputs(),
                         [a * b for (a, b) in zip([1, 2, 3, 4, 5], [i ** 2 for i in [1, 2, 3, 4, 5]])])

        # should not process it again
        job_final.result = []
        job_final.process()
        self.assertEqual(job_final.get_outputs(), [])
        self.assertNotEqual(job_final.args, [])
