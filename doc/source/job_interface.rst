=================
The Job Interface
=================

This page describes the Job interface that is used for building a workflow.

------------
Introduction
------------

A Job is a Node in a Directed Acyclic Graph that defines an action on the output of its parents and specifies output(s) that other Jobs/Nodes can use again as their inputs.

.. note::
    Root Jobs of the DAG only operate on specified parameters and not
    on inputs of other Jobs.

Each Job may have parameters that define its current state. These parameters are flushed to a JSON file together with the outputs of the Job. This allows us to
store intermediate computations in a simple manner and to avoid recomputing actions whose parameters did not change.

All this functionality (loading/storing state, comparing states, check for outdated parents) is taken care of by the :py:class:`Job <SourceCode.video.processing.job.Job>` class.
It also initializes all specified parameters so we only have to specify these via name (See :ref:`example_job` for the implementation of a Job example)

Subclasses only need to define their specific parameters, outputs and parents and overload the :py:func:`run <SourceCode.video.processing.job.Job.run>` and
:py:func:`get_outputs <SourceCode.video.processing.job.Job.get_outputs>` functions.

If the default method for loading the state back from the JSON file needs some additional functionality, it is possible to overload the
:py:func:`load_state <SourceCode.video.processing.job.Job.load_state>` function.



.. _example_job:

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

This specific Job has no special parameters, if it had, we would set those in `attributes_to_serialize`. ::

    attributes_to_serialize = []

Next we need to specify what outputs are cached in the JSON file (This automatically sets an attribute with the name *histogram_correlations* for this Job). ::

    outputs_to_cache = ['histogram_correlations']

The `__init__` method only calls the Job's `__init__` method. If there were any special parameters for this Job
we could assert that those were actually passed. ::

    def __init__(self, *args, **kwargs):
        super(HistogramCorrelationJob, self).__init__(*args, **kwargs)

        # Optional
        # assert('name_of_job_parameter' in kwargs)

The next optional thing to do is to overload the `load_state` function. It is responsible for loading the
stored state back from the JSON file.

We need to overload this function, when the JSON storage differs from the format we want to have in Python.
JSON stores all keys as Strings for example, and for this specific Job we want to index the stored data by
integers and also sort it, so we do ::

    from ....util.tools import sort_dictionary_by_integer_key

    [...]

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

The action of the Job is defined in the :py:func:`run <SourceCode.video.processing.job.Job.run>` method which every Job needs to overload.

##############
Parent Inputs
##############

The :py:func:`run <SourceCode.video.processing.job.Job.run>` method receives its argumentes in the same order that the parents of the Job are
specified.

When building a workflow we can for example specify ::

    HistogramCorrelationJob.add_parent(HistogramsLABDiff)
    HistogramCorrelationJob.add_parent(NumberOfFilesJob)

or alternatively directly set the `parents` member of the Job ::

    parents = [HistogramsLABDiff, NumberOfFilesJob]


The first parent returns a function of `time` and `area_name` (see :py:class:`HistogramsLABDiff <SourceCode.video.processing.jobs.histogram_computation.HistogramsLABDiff>`)
and the second parent just returns the number of thumbnails. So the :py:func:`run <SourceCode.video.processing.job.Job.run>` method looks as follows ::

    def run(self, *args, **kwargs):

        # The first parent is the HistogramComputation
        get_histogram = args[0]

        # Second parent is the NumberOfFiles
        number_of_files = args[1]

        # init
        self.histogram_correlations = {}

        previous_slide_histogram = get_histogram('slides', 1)
        previous_speaker_histogram_plane = get_speaker_histogram_plane(1)

        for frame_index in range(2, number_of_files):

            slide_histogram = get_histogram('slides', frame_index)

            if previous_slide_histogram is not None:
                self.histogram_correlations[frame_index] = \
                    cv2.compareHist(slide_histogram, previous_slide_histogram, cv2.cv.CV_COMP_CORREL)

            previous_slide_histogram = slide_histogram
            previous_speaker_histogram_plane = speaker_histogram_plane


The :py:func:`run <SourceCode.video.processing.job.Job.run>` method builds up a dictionary with the indices being the frames and the
values being the corresponding histogram correlation.

This dictionary is saved to the JSON file as specified by the `outputs_to_cache` member of
:py:class:`HistogramCorrelationJob <SourceCode.video.processing.jobs.histogram_correlations.HistogramCorrelationJob>`.


##############
Job Outputs
##############

Since it is more convenient for other Jobs to operate with a proper function instead of just a dictionary we transform the output in the
:py:func:`get_outputs <SourceCode.video.processing.job.Job.get_outputs>` method. ::

    def get_outputs(self):
        super(HistogramCorrelationJob, self).get_outputs()

        if self.histogram_correlations is None:
            raise RuntimeError('The Correlations between the histograms have not been computed yet.')

        return Functor(self.histogram_correlations)

The :py:class:`Functor <SourceCode.util.functor.Functor>` class just wraps the dictionary in a callable object.
When this object is called with an index, it returns the value of the index from the dictionary.

---------
Reference
---------

.. automodule:: SourceCode.video.processing.job
   :members:
   :undoc-members:
