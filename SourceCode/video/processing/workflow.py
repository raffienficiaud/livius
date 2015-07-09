
from .video.processing.job import Job
from .ffmpeg_to_thumbnails import factory as ffmpeg_factory

def workflow():
    """Returns a workflow definition. The processing of the workflow is performed in another module"""
    
    
    ffmpeg = ffmpeg_factory()
    
    return ffmpeg

