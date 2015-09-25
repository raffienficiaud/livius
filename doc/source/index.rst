.. Livius documentation master file, created by
   sphinx-quickstart on Wed Jul 29 13:42:33 2015.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Livius's documentation!
==================================

Contents:

.. toctree::
   :maxdepth: 2

   Existing Jobs<jobs>
   Existing Workflows<workflow>
   Output video creation<final_video>
   job_interface
   example_workflow
   utilities
   sphinx_how_to

Getting started
===============

Livius defines `workflow` and `jobs`: those are convenient tools for being able to process the videos.

* A `job` is a small processing unit that takes input and produces outputs, and maintains a state so that
  outputs are not being reprocessed if the inputs have not changed.
* A `workflow` is a set of jobs with their dependencies (which defines a Directed Acyclic Graph)

Livius already contains many different jobs as well as some workflows. 

* The available jobs are detailed here: :doc:`jobs`
* The available workflows are detailed here: :doc:`workflow`
* A workflow example is detailed here: :doc:`example_workflow`
* It is also easy to define new jobs: the job interface is detailed here: :doc:`job_interface`

Running Livius
==============
Once you know more about the workflow you want to use, you may run the processing of this 
workflow over a bunch of videos simply by running

.. code::

  cd $livius_src
  python -m SourceCode --workflow=my_workflow --output-folder=/a/big/disk --video-folder=my_folder_full_of_videos

The detailed options of Livius are given below (just type ``--help`` in the previous command line):

.. program-output:: python -m SourceCode --help
    :cwd: ../../ 


Installation
============

For FFMpeg related processing, ``ffmpeg`` should be installed and accessible from the command line. The 
dependencies of Livius are the following (which may be installed in a virtual environment, see/search
in the wiki for more details).

.. code::

  pip install numpy
  pip install moviepy
  
  # for the documentation
  pip install sphinx
  pip install sphinxcontrib-programoutput 
  pip install sphinx_bootstrap_theme

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
