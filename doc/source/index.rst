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

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
