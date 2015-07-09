
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
        return [0,1,2,3,4]
    
class Job2(Job):
    name = 'job2'
    
    def run(self):
        pass
    
    def get_outputs(self):
        return [1,2,3,4,5]

def process_workflow(final_job_cls, **kwargs):
    
    # creation of the processing graph
    final_job_inst = final_job_cls(**kwargs)
    
    final_job_inst.get_outputs()


class JobTests(unittest.TestCase):
    
    
    def setUp(self):
        self.assertIsNone(Job1.parent_tasks)
        self.assertIsNone(Job2.parent_tasks)
        
        self.tmpdir = mkdtemp()
        pass
    
    def tearDown(self):
        Job1.parent_tasks = None
        Job2.parent_tasks = None
        
        shutil.rmtree(self.tmpdir)
    
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
        
    def test_parent_json(self):
        
        Job2.add_parent(Job1)
        job_final = Job2(json_prefix=os.path.join(self.tmpdir, 'toto_testp'))
        job_final.serialize_state()
        
        self.assertTrue(os.path.exists(os.path.join(self.tmpdir, 'toto_testp_job1.json')))
        self.assertTrue(os.path.exists(os.path.join(self.tmpdir, 'toto_testp_job2.json')))