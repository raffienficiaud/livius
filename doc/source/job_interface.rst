=================
The Job Interface
=================

This page describes the Job interface that is used for building a workflow.

------------
Introduction
------------

A Job is a Node in a Directed Acyclic Graph that performs an **action** on values coming into the Job (the output of its parents) and 
specifies output(s) that other Jobs/Nodes can use again as their inputs.
In the following, **upstream** means that the values are being consumed by the Job and coming from the parent job(s), while **downstream** means that the job is providing
outputs to the child jobs (consumed or not).  

.. note::
    Root Jobs of the DAG only operate on specified parameters and do not 
    use any other upstream inputs. 

---------------------------------
Job state and computation caching
---------------------------------

Jobs are specially designed to avoid recomputations. They store their state and their outputs to the file system. For the purpose of storing their state, Jobs provide the 
functions to ease the definition of what is needed and what should be stored. 
When a Job is asked to perform the **action** that it is supposed to do, it first checks if the parents need to recompute anything. If not, it means that the parents 
state can be immediately retrieved from the stored file, which saves the computations at the cost of loading/saving the state. 
For the current job, if the parameters have not changed and the parents do not need to recomputed their output,
the current job may also retrieve its state from the stored file.

All this functionality (loading/storing state, comparing states, check for outdated parents) is already taken care of by the 
:py:class:`Job <livius.video.processing.job.Job>` class, and only a few number of *fields* need to be set up. 

*********
Extension
*********
If the default method for loading the state back from the JSON file needs some additional functionality, it is possible to overload the
:py:func:`Job.load_state <livius.video.processing.job.Job.load_state>` function.

*****************************
Runtime and static parameters
*****************************
Each Job may have parameters that define its current state. These parameters are flushed to a JSON file together with the outputs of the Job. This allows us to
store intermediate computations in a simple manner and to avoid recomputing actions whose parameters did not change.

It also initializes all specified parameters so we only have to specify these via name (See :ref:`example_job` for the implementation of a Job example)


------------------
Job identification
------------------

Every job is named using the simple ``name`` attribute::

    class HistogramCorrelationJob(Job):
        name = 'histogram_correlation'
        
This name **should** be unique in the workflow. If the behaviour of a job is needed several times in a workflow (but with different parameters), then 
new classes may be defined being child of the job to reuse, this time with a different name. This is the case for instance for the classes 
:py:class:`SelectSlide <livius.video.processing.jobs.select_polygon.SelectSlide>` and :py:class:`SelectSpeaker <livius.video.processing.jobs.select_polygon.SelectSpeaker>`, 
refining the behaviour of :py:class:`SelectPolygonJob <livius.video.processing.jobs.select_polygon.SelectPolygonJob>`.

--------------------
Job cache comparison
--------------------
The main entry of the cached value comparison is performed by the method :py:func:`Job.is_up_to_date <livius.video.processing.job.Job.is_up_to_date>`, which is possible
to override in a child class (for eg. test for file existance, timestamp of files, etc).

This method checks that the parents are up to date (and returns ``False`` if not), and then calls the function
:py:func:`Job.are_states_equal <livius.video.processing.job.Job.are_states_equal>`, which is also possible to override.

The current implementation of the comparison is that the values being compared are transformed to a string, and the 
resulting strings are compared instead. 

.. note::

   If you override one of those member function, it is always possible to fall-back/call the default behaviour with
   for instance::
   
       super(CurrentJobClass, self).is_up_to_date() 

----------
Job action
----------

Subclasses only need to define their specific parameters, outputs and parents and overload the :py:func:`Job.run <livius.video.processing.job.Job.run>` and
:py:func:`Job.get_outputs <livius.video.processing.job.Job.get_outputs>` functions.



.. _example_job:

-----------
Example Job
-----------

We are going to use the :py:class:`HistogramCorrelationJob <livius.video.processing.jobs.histogram_correlations.HistogramCorrelationJob>` to explain
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

The action of the Job is defined in the :py:func:`run <livius.video.processing.job.Job.run>` method which every Job needs to overload.

##############
Parent Inputs
##############

The :py:func:`run <livius.video.processing.job.Job.run>` method receives its argumentes in the same order that the parents of the Job are
specified.

When building a workflow we can for example specify ::

    HistogramCorrelationJob.add_parent(HistogramsLABDiff)
    HistogramCorrelationJob.add_parent(NumberOfFilesJob)

or alternatively directly set the `parents` member of the Job ::

    parents = [HistogramsLABDiff, NumberOfFilesJob]


The first parent returns a function of `time` and `area_name` (see :py:class:`HistogramsLABDiff <livius.video.processing.jobs.histogram_computation.HistogramsLABDiff>`)
and the second parent just returns the number of thumbnails. So the :py:func:`run <livius.video.processing.job.Job.run>` method looks as follows ::

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


The :py:func:`run <livius.video.processing.job.Job.run>` method builds up a dictionary with the indices being the frames and the
values being the corresponding histogram correlation.

This dictionary is saved to the JSON file as specified by the `outputs_to_cache` member of
:py:class:`HistogramCorrelationJob <livius.video.processing.jobs.histogram_correlations.HistogramCorrelationJob>`.


##############
Job Outputs
##############

Since it is more convenient for other Jobs to operate with a proper function instead of just a dictionary we transform the output in the
:py:func:`get_outputs <livius.video.processing.job.Job.get_outputs>` method. ::

    def get_outputs(self):
        super(HistogramCorrelationJob, self).get_outputs()

        if self.histogram_correlations is None:
            raise RuntimeError('The Correlations between the histograms have not been computed yet.')

        return Functor(self.histogram_correlations)

The :py:class:`Functor <livius.util.functor.Functor>` class just wraps the dictionary in a callable object.
When this object is called with an index, it returns the value of the index from the dictionary.

---------
Reference
---------

.. automodule:: livius.video.processing.job
   :members:
   :undoc-members:
