

"""This file contains the "preparation" functions from the video, such as the 
  
  - creation of the thumnail images
  - computation of the min/max/histograms of the appropriate regions in frames
  
"""

import os
import json
import logging


logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


class Job(object):
    
    name = "root"
    attributes_to_serialize = []
    parent_tasks = None
    
    @classmethod
    def add_parent(cls, obj):
        
        if obj is None:
            return
        
        if not issubclass(obj, Job):
            logger.error("Adding a erroneous class definition to the graph: %r not a subclass of %r", obj, cls)
            raise RuntimeError("Adding a erroneous class definition to the graph")
        
        if cls.parent_tasks is None:
            cls.parent_tasks = []
        cls.parent_tasks.append(obj)
        
    @classmethod
    def get_parents(cls):
        return cls.parent_tasks
    
    def __init__(self, **kwargs):
        
        json_prefix = kwargs.get('json_prefix', '')
        self.json_filename = json_prefix + '_' + self.name + '.json'
    
        # creation of the serialized attributes
        for k in self.attributes_to_serialize:
            setattr(self, k, None) 
        
        for name, value in kwargs.items():
            setattr(self, name, value)

        self.parent_instances = []
        if self.parent_tasks is not None:
            for par in self.parent_tasks:
                self.parent_instances.append(par(**kwargs))
            
        
    def is_up_to_date(self):
        """Contains the logic to indicate that this step should be processed again"""
        for par in self.parent_instances:
            if not par.is_up_to_date():
                return False
        return True
    
    def are_states_equal(self):
        """Returns True is the state of the current object is the same as the one in the serialized json dump"""
        if self.json_filename is None:
            return False
        
        if not os.path.exists(self.json_filename):
            return False
        
        dict_json = json.load(open(self.json_filename))
        
        try:
            for k in self.attributes_to_serialize:
                if str(dict_json[k]) != str(getattr(self, k)):
                    logger.debug("Key %s mismatch: left=%s / right=%s", k, str(dict_json[k]), str(getattr(self, k)))
                    return False
        
        except KeyError:
            return False
        
        except Exception, e:
            logger.debug("Exception caught in 'are_states_equal: %r", e)
            return False
        
        return True
        
    
    def serialize_state(self):
        """Flushes the state of the runner into the json file mentioned by self.json_filename"""
        
        if self.json_filename is None:
            return
        
        d = {}
        for k in self.attributes_to_serialize:
            d[k] = getattr(self, k)
        
        with open(self.json_filename, 'w') as f:
            json.dump(d, f)
        
    
    def run(self):
        """Runs the action and all parents actions"""
        pass
    

    def get_outputs(self):
        """Returns all the possible outputs of this step"""
        if not self.is_up_to_date():
            raise RuntimeError("Cannot query for the outputs before those are computed")
        
        return None


        

def prepare(video_filename):
    
    
    obj_extract = FFMpegThumbnailsJob(video_filename=video_filename,
                                      video_width=640,
                                      video_fps=1)
    
    if not obj_extract.is_run_up_to_date():
        obj_extract.run()



if __name__ == '__main__':

    storage = '/media/renficiaud/linux-data/'
    filename = 'BlackmagicProductionCamera 4K_1_2015-01-16_1411_C0000.mov'
    
    prepare(os.path.join(storage, filename))
    
