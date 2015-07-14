
import logging
FORMAT = '[%(asctime)-15s] %(message)s'
logging.basicConfig(format=FORMAT)

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


from .job import Job
from .ffmpeg_to_thumbnails import factory as ffmpeg_factory


def workflow():
    """Returns a workflow definition. The processing of the workflow is performed in another module"""
    
    
    ffmpeg = ffmpeg_factory()
    
    return ffmpeg


def process(workflow_instance, **kwargs):
    
    instance = workflow_instance(**kwargs)
    instance.process()

    out = instance.get_outputs()
    instance.serialize_state()

    return out


if __name__ == '__main__':
    
    import os
    from tempfile import mkdtemp
    tmpdir = mkdtemp()
    
    d = dict([('video_filename', 
              os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir, "Videos", "video_7.mp4")
              )]) 
    
    current_workflow = workflow()
    outputs = process(current_workflow, 
                      json_prefix=os.path.join(tmpdir, 'test_video7'),
                      **d)
    
    print outputs
    
    