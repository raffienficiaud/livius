======================
Implemented Jobs
======================

Introduction
------------
Several jobs are already implementation and may be used for creating new workflows.
Each job being a node in the workflow DAG, the documentation below tells:

* what are the parameters at runtime (set by the command line interface)
* what inputs are expected from the parent nodes
* what outputs are produced

Categories
----------
The jobs fall into the following categories:

* user interaction: this is meanly for user selection of image parts. This category
  is covered by the :py:mod:`select_polygon <SourceCode.video.processing.jobs.select_polygon>`
  module.

* slide processing: the functionality is covered by the module 
  :py:mod:`extract_slide_clip <SourceCode.video.processing.jobs.extract_slide_clip>`
  
* video content creation: those type of jobs create objects suitable for MoviePy as inputs. Some utility
  classes/functions are also provided in the :py:mod:`dummy_clip <SourceCode.video.processing.jobs.dummy_clip>`
  module. 
  
* metadata management: jobs that are able to provide the workflow with meta data that will be used 
  to eg. describe the video, add content (introduction, epilog images), etc. Located in the module 
  :py:mod:`meta <SourceCode.video.processing.jobs.meta>`

Reference
---------

.. toctree::
   :name: jobtoc

   video.processing.jobs.select_polygon<jobs/select_polygon>
   video.processing.jobs.contrast_enhancement_boundaries<jobs/contrast_enhancement>
   video.processing.jobs.extract_slide_clip<jobs/extract_slide_clip>
   video.processing.jobs.ffmpeg_to_thumbnails<jobs/ffmpeg_to_thumbnails>
   video.processing.jobs.histogram_computation<jobs/histogram_computation>
   video.processing.jobs.histogram_correlations<jobs/histogram_correlations>
   video.processing.jobs.segment_computation<jobs/segment_computation>
   video.processing.jobs.create_movie<jobs/video_content_creation>
   video.processing.jobs.dummy_clip<jobs/dummy_clips>
   video.processing.jobs.dummy_clip<jobs/metadata>
