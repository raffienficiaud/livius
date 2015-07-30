=================
The Job Interface
=================

------------
Introduction
------------

This file describes the Job interface that is used for building a workflow.


-----------
Example Job
-----------

We are going to use the :py:class:`HistogramCorrelationJob <SourceCode.video.processing.jobs.histogram_correlations.HistogramCorrelationJob>` to explain
how to build further Jobs.

*********
Job Setup
*********

We first need to setup a Job by subclassing it and setting a name::

    class HistogramCorrelationJob(Job):
    name = 'histogram_correlation'

This specific Job has no special parameters, we would set those in `attributes_to_serialize`. ::

    attributes_to_serialize = []

Next we need to specify what outputs are cached in the JSON file. ::

    outputs_to_cache = ['histogram_correlations']

The `__init__` method only calls the Job's `__init__` method. If there were any parameters we could
assert that those were actually passed. ::

    def __init__(self, *args, **kwargs):
        super(HistogramCorrelationJob, self).__init__(*args, **kwargs)

        # Optional
        # assert('name_of_job_parameter' in kwargs)

The next optional thing to do is to overload the `load_state` function. It is responsible for loading the
stored state back from the JSON file.

We need to overload this function, when the JSON storage differs from the format we want to have in Python.
JSON stores all keys as Strings for example, and for this specific Job we want to index the stored data by
integers and also sort it, so we do ::

    def load_state(self):
        state = super(HistogramCorrelationJob, self).load_state()

        if state is None:
            return None

        correlations = state['histogram_correlations']
        correlations = sort_dictionary_by_integer_key(correlations)

        state['histogram_correlations'] = correlations

        return state


***********************
Defining the Job Action
***********************

The action of the Job is defined in the `run` method.

##############
Parent Inputs
##############



##############
Job Outputs
##############

---------
Reference
---------

.. automodule:: SourceCode.video.processing.job
   :members:
   :undoc-members:
