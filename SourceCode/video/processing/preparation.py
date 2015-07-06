

"""This file contains the "preparation" functions from the video, such as the 
  
  - creation of the thumnail images
  - computation of the min/max/histograms of the appropriate regions in frames
  
"""

import os
import time
import json
import subprocess


class Preparation(object):
    
    name = "dummy_prepare"
    filename = name + '.json'
    
    def __init__(self, *args, **kwargs):
        
        for name, value in kwargs.items():
            setattr(self, name, value)
    
    def is_preparation_needed(self):
        return False
     
    def action_video(self, video_filename):
        """Runs an action on the video file"""
        pass
    
    def action_frame(self, frame, time_frame):
        """Runs an action on the provided frame"""
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
    
    def __init__(self, *args, **kwargs):
        super(self, ExtractThumbnails).__init__(*args, **kwargs)


    def action_video(self, video_file_name):
        thumbnail_path = os.path.join(os.path.dirname(video_file_name), 'thumbnails')
        if not os.path.exists(thumbnail_path):
            os.makedirs(thumbnail_path)

        extract_thumbnails(video_file_name=video_file_name, 
                           output_width=640, 
                           output_folder=thumbnail_path)





def prepare(video_file_name):
    
    
    file_preprocessing = os.path.splitext(video_file_name)[0] + '.json'
    if not file_preprocessing:
        prepare_dict = {}
        prepare_dict['thumbnails_extracted'] = False
    else:
        with open(file_preprocessing) as f:
            prepare_dict = json.load(f)
             
            
    # should we extract the thumbnails
    do_extract_thumbnails = not prepare_dict['thumbnails_extracted']
    if do_extract_thumbnails:
        thumbnail_path = os.path.join(os.path.dirname(video_file_name), 'thumbnails')
        if not os.path.exists(thumbnail_path):
            os.makedirs(thumbnail_path)
        extract_thumbnails(video_file_name, 
                           output_folder=thumbnail_path)
        prepare_dict['thumbnails_extracted'] = True
        
        with open(file_preprocessing, 'w') as f:
            json.dump(prepare_dict, f)
            
    