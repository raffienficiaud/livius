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

Reference
---------

.. automodule:: SourceCode.video.processing.jobs.select_polygon
   :members:
   :special-members:

.. automodule:: SourceCode.video.processing.jobs.contrast_enhancement_boundaries
   :members:
   :special-members:

.. automodule:: SourceCode.video.processing.jobs.extract_slide_clip
   :members:
   :special-members:

.. automodule:: SourceCode.video.processing.jobs.ffmpeg_to_thumbnails
   :members:
   :special-members:

.. automodule:: SourceCode.video.processing.jobs.histogram_computation
   :members:
   :special-members:

.. automodule:: SourceCode.video.processing.jobs.histogram_correlations
   :members:
   :special-members:

.. automodule:: SourceCode.video.processing.jobs.segment_computation
   :members:
   :special-members:
