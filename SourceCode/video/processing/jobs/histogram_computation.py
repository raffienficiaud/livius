"""
This file provides the Job interface for the computation of the several histograms in
order to detect different changes in the scene (lightning, speaker motion, etc).
"""

from ..job import Job
import os
import json
import cv2
import numpy as np



class HistogramsCorrelation(Job):
    """
    Computes histogram correlation between two consecutive frames in a specific area of the plane (normalized coordinates).
    
    Expect as parents, in this order:
    
    - a rectangle specifying the location where the histogram should be computed
    - a list of images
    - potentially a inside the rectangle to remove (this location will be narrowed to the rectangle)
    
    The output is:
    - a function of frame index (one argument) that provides the histogram correlation
    
    The state of this function is saved on the json file.
    
    """
    
    
    name = 'histogram_lab_space'
    attributes_to_serialize = ['rectangle_location',
                               'rectangle_to_remove',
                               'number_of_files',
                               'histogram_correlations'
                               ]
    
    
    def __init__(self,
                 *args,
                 **kwargs):
        """
        :param :
        """
        super(HistogramsCorrelation, self).__init__(*args, **kwargs)
        
        self._get_previous_state()
    
        # read back the output files if any
        

    def _get_previous_state(self):
        if not os.path.exists(self.json_filename):
            return None

        with open(self.json_filename) as f:
            d = json.load(f)

            if 'histogram_correlations' in d:
                self.histogram_correlations = d['histogram_correlations']

            if 'number_of_files' in d:
                self.number_of_files = self._get_correlations() 

    def _get_correlations(self):
        if not os.path.exists(self.json_filename):
            return None

        with open(self.json_filename) as f:
            d = json.load(f)
            if 'histogram_correlations' not in d:
                return None
            return d['histogram_correlations']    
    
    
    def is_up_to_date(self):
        """Returns False if no correlation has been computed (or can be restored from 
        the json dump), default behaviour otherwise"""
        if not self.histogram_correlations:
            return False

        return super(HistogramsCorrelation, self).is_up_to_date()    
    
    def run(self, *args, **kwargs):
        assert(len(args) >= 2)
        
        self.rectangle_location = args[0]
        self.rectangle_to_remove = args[2] if len(args) > 2 else None

        image_list = args[1]
        self.number_of_files = len(image_list)
        
        
        # do the computation
        self.histogram_correlations = []
        
        # save the state (commit)
        
        self.serialize_state()