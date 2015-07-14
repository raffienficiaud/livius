
import unittest
from tempfile import mkdtemp
import os
import shutil

# logging facility
import logging
FORMAT = '[%(asctime)-15s] %(message)s'
logging.basicConfig(format=FORMAT)

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

from ..video.processing.job import Job


class Job1(Job):
    name = 'job1'

    def get_outputs(self):
        return [1, 2, 3, 4, 5]


class Job2(Job):
    name = 'job2'

    def run(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        pass

    def get_outputs(self):
        return [1, 2, 3, 4, 5]


class JobTestsFixture(object):
    """Fixture for cleaning Job1/Job2 and the temporary directory"""

    def setUp(self):
        self.assertIsNone(Job1.parents)
        self.assertIsNone(Job2.parents)
        self.assertFalse(Job1.attributes_to_serialize)
        self.assertFalse(Job2.attributes_to_serialize)

        self.tmpdir = mkdtemp()
        pass

    def tearDown(self):
        Job1.parents = None
        Job2.parents = None
        Job1.attributes_to_serialize = []
        Job2.attributes_to_serialize = []

        shutil.rmtree(self.tmpdir)


class JobTests(JobTestsFixture, unittest.TestCase):

    def test_json_prefix(self):
        job = Job1(json_prefix=os.path.join(self.tmpdir, 'toto'))
        job.serialize_state()

        self.assertTrue(os.path.exists(os.path.join(self.tmpdir, 'toto_job1.json')))

    def test_are_state_equal(self):
        class JobT(Job):
            name = 'jobT'
            attributes_to_serialize = ['bread', 'cucumbers']

        job = JobT(json_prefix=os.path.join(self.tmpdir, 'toto_test'))

        # not state yet
        self.assertFalse(job.are_states_equal())
        job.serialize_state()

        job2 = JobT(json_prefix=os.path.join(self.tmpdir, 'toto_test'))
        self.assertTrue(job2.are_states_equal())

        job3 = JobT(json_prefix=os.path.join(self.tmpdir, 'toto_test'))
        job3.cucumbers = 10
        self.assertFalse(job3.are_states_equal())

    def test_parents(self):

        Job2.add_parent(Job1)
        self.assertIn(Job1, Job2.get_parents())
        self.assertIsNone(Job1.get_parents())

    def test_get_parent(self):

        Job2.add_parent(Job1)
        job_final = Job2(json_prefix=os.path.join(self.tmpdir, 'toto_test_parent'))
        job1_instance = job_final.get_parent_by_type(Job1)
        self.assertIsNotNone(job1_instance)
        self.assertIsInstance(job1_instance, Job1)

        self.assertIsNone(job_final.get_parent_by_type(Job2))
        self.assertIsNone(job1_instance.get_parent_by_type(Job1))

        job1_instance_bis = job_final.get_parent_by_name('job1')
        self.assertIs(job1_instance, job1_instance_bis)

        self.assertIsNone(job_final.get_parent_by_name('jobx'))
        self.assertIsNone(job1_instance.get_parent_by_name('job1'))
        self.assertIsNone(job1_instance.get_parent_by_name('jobx'))

    def test_parent_json(self):
        Job2.add_parent(Job1)
        job_final = Job2(json_prefix=os.path.join(self.tmpdir, 'toto_testp'))
        job_final.serialize_state()

        self.assertTrue(os.path.exists(os.path.join(self.tmpdir, 'toto_testp_job1.json')))
        self.assertTrue(os.path.exists(os.path.join(self.tmpdir, 'toto_testp_job2.json')))

    def test_parent_up_to_date(self):
        Job1.attributes_to_serialize.append('job1attr1')
        Job1.attributes_to_serialize.append('job1attr2')
        Job2.attributes_to_serialize.append('job2attr1')

        Job2.add_parent(Job1)

        job_final = Job2(json_prefix=os.path.join(self.tmpdir, 'toto_test_uptodate'))

        self.assertFalse(job_final.is_up_to_date())

        job_final.serialize_state()
        self.assertTrue(job_final.is_up_to_date())

        job1_instance = job_final.get_parent_by_type(Job1)
        self.assertTrue(hasattr(job1_instance, "job1attr1"))
        job1_instance.job1attr1 = 10
        self.assertFalse(job_final.is_up_to_date())
        job_final.serialize_state()
        self.assertTrue(job_final.is_up_to_date())


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


class JobParentAttributeTests(JobTestsFixture, unittest.TestCase):
    """Tests the access to the parent Jobs through a simple API"""

    def setUp(self):
        super(JobParentAttributeTests, self).setUp()
        Job1.attributes_to_serialize.append('job1attr1')
        Job1.attributes_to_serialize.append('job1attr2')
        Job2.attributes_to_serialize.append('job2attr1')
        Job2.add_parent(Job1)

    def test_metaclass_parent_access(self):
        job_final = Job2(json_prefix=os.path.join(self.tmpdir, 'test_meta'))
        self.assertTrue(hasattr(job_final, Job1.name))
        self.assertIsNone(job_final.job1.job1attr1)
        self.assertIs(job_final.job1, job_final.get_parent_by_name('job1'))

    def test_cannot_assign_parent(self):
        job_final = Job2(json_prefix=os.path.join(self.tmpdir, 'test_meta'))
        import exceptions
        with self.assertRaises(exceptions.AttributeError):
            job_final.job1 = Job1()

    def test_cannot_add_twice_same_name_parent(self):
        class Job1bis(Job1):
            attributes_to_serialize = Job1.attributes_to_serialize + ['test']
            pass

        Job2.add_parent(Job1bis)

        with self.assertRaises(RuntimeError):
            Job2(json_prefix=os.path.join(self.tmpdir, 'test_meta'))
