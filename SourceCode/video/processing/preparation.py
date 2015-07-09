

"""This file contains the "preparation" functions from the video, such as the 
  
  - creation of the thumnail images
  - computation of the min/max/histograms of the appropriate regions in frames
  
"""

import os
import time
import json
import subprocess
import logging

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


class Preparation(object):
    
    name = "dummy_prepare"
    attributes_to_serialize = []
        
    def __init__(self, *args, **kwargs):
        
        for name, value in kwargs.items():
            setattr(self, name, value)
            
        if 'json_filename' not in kwargs:
            self.json_filename = None
    
    def is_run_up_to_date(self):
        """Indicates that this step should be processed again"""
        return False
    
    def are_states_equal(self):
        """Returns True is the state of the current object is the same as the one in the serialized json dump"""
        if self.json_filename is None:
            return False
        
        if not os.path.exists(self.json_filename):
            return False
        
        dict_json = json.load(open(self.json_filename))
        
        try:
            for k in self.attributes_to_serialize:
                if int(dict_json[k]) != str(getattr(self, k)):
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
        """Runs the action"""
        pass
    


def extract_thumbnails(video_file_name, output_width, output_folder):
    """Extracts the thumbnails using FFMpeg.
    
    :param video_file_name: name of the video file to process
    :param output_width: width of the resized images
    :param output_folder: folder where the thumbnails are stored
    
    """
    args = ['ffmpeg', '-i', os.path.abspath(video_file_name), '-r', '1', '-vf', 'scale=%d:-1' % output_width, '-f', 'image2',  '%s/frame-%05d.png' % os.path.abspath(output_folder)]
    proc = subprocess.Popen(args)
    
    return_code = proc.poll() 
    while return_code is None:
        time.sleep(1)
        return_code = proc.poll()
        
    return


class ExtractThumbnails(Preparation):
    
    name = "thumbnails_generation"
    attributes_to_serialize = ['video_filename', 'video_fps', 'video_width', 'output_location']
    
    def __init__(self, 
                 video_filename,
                 video_width = None,
                 video_fps = None, 
                 output_location = None,
                 *args, 
                 **kwargs):
        """
        :param output_location: absolute location of the generated thumbnails
        """
        
        super(self, ExtractThumbnails).__init__(*args, **kwargs)
        
        if video_filename is None:
            raise RuntimeError("The video file name cannot be empty")
        
        if video_width is None:
            video_width = 640

        if video_fps is None:
            video_fps = 1
            
        if output_location is None:
            output_location = os.path.join(os.path.dirname(video_filename), 'thumbnails')
        
        self.video_filename = os.path.abspath(video_filename)
        self.video_fps = video_fps
        self.video_width = video_width
        self.output_location = output_location
        self.json_filename = os.path.splitext(video_filename)[0] + '_' + self.name + '.json'

    def is_run_up_to_date(self):
        
        if not os.path.exists(self.output_location):
            return False
        
        return self.are_states_equal()

    def run(self):
        if not os.path.exists(self.output_location):
            os.makedirs(self.output_location)

        extract_thumbnails(video_file_name=self.video_filename, 
                           output_width=self.video_width, 
                           output_folder=self.output_location)

        # commit to the json dump
        self.serialize_state()
        

def prepare(video_filename):
    
    
    obj_extract = ExtractThumbnails(video_filename=video_filename,
                                    video_width=640,
                                    video_fps=1)
    
    if not obj_extract.is_run_up_to_date():
        obj_extract.run()



if __name__ == '__main__':

    storage = '/media/renficiaud/linux-data/'
    filename = 'BlackmagicProductionCamera 4K_1_2015-01-16_1411_C0000.mov'
    
    prepare(os.path.join(storage, filename))
    
