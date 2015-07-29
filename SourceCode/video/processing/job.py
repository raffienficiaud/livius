"""
This module provides a Job class that all Actions should subclass.
==================================================================

The :class:`Job` class takes care of the dependencies between Jobs and only runs
computations if needed. This can happen if some parameters of a predecessing action change.
"""

import os
import json
import logging
from exceptions import AttributeError

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


class Job(object):

    """
    This class implements the basic functionality for each Job.


    All subclasses must override the run() function. It should compute all attributes
    mentioned in self.outputs_to_cache.

    After the Job is run, the state is serialized to a JSON file.

    Subclasses can optionally override the load_state() function which provides a way to
    deal with the difference between JSON storage and the Python objects
    (e.g the fact that keys are always stored as unicode strings).
    """

    #: Name of the Job (used for identification).
    name = "root"

    #: List of attributes that represent the Job's state
    attributes_to_serialize = []

    #: Outputs of the Job.
    outputs_to_cache = []

    #: List of parents.
    #:
    #: .. important::
    #:      The order of the parents is important as it
    #:      determines the order in which the outputs of
    #:      outputs of the parent Jobs are passed to this
    #:      Job's :func:`run` method.
    parents = None

    # private API
    _is_frozen = False

    @classmethod
    def add_parent(cls, obj):
        """
        Add a specific job as a parent job.

        Parent Jobs are jobs which the current job is dependent on. They are
        executed and updated before the current job.
        """
        if obj is None:
            return

        if not issubclass(obj, Job):
            logger.error("Adding a erroneous class definition to the graph: %r not a subclass of %r", obj, cls)
            raise RuntimeError("Adding a erroneous class definition to the graph")

        if cls.parents is None:
            cls.parents = []
        cls.parents.append(obj)

    @classmethod
    def get_parents(cls):
        """Return all the parents jobs of this class."""
        return cls.parents

    def __init__(self, *args, **kwargs):
        """:param json_prefix: the prefix used for serializing the state of this runner."""
        super(Job, self).__init__()

        self._is_frozen = False

        json_prefix = kwargs.get('json_prefix', '')
        self.json_filename = json_prefix + '_' + self.name + '.json'

        # creation of the serialized attributes
        for k in self.attributes_to_serialize:
            setattr(self, k, None)

        for k in self.outputs_to_cache:
            setattr(self, k, None)

        for name, value in kwargs.items():
            setattr(self, name, value)

        # Creation of all parents
        self._parent_names = []
        self._parent_instances = []

        self._predecessor_instances = {}

        if self.parents is not None:
            for par in self.parents:

                # Look for all predecessors that have already been constructed
                predecessors_in_kwargs = kwargs.get('predecessors', {})

                if par.name in predecessors_in_kwargs:
                    # Parent name has already been constructed. Use this instance
                    par_instance = predecessors_in_kwargs[par.name]
                else:
                    # We need to construct a new instance of this parent
                    par_instance = par(*args, **kwargs)

                # Update the predecessors with the predecessors of this parent
                predecessors = par_instance._predecessor_instances
                predecessors_in_kwargs.update(predecessors)
                kwargs['predecessors'] = predecessors_in_kwargs
                self._predecessor_instances.update(predecessors)

                if par_instance.name in self._parent_names:
                    # not adding
                    raise RuntimeError("name %s is already used for one parent of job %s" %
                                       (par_instance.name, self.name))

                self._parent_names.append(par_instance.name)
                self._parent_instances.append(par_instance)
                setattr(self, par.name, par_instance)

        # This Job is now completely initialized.
        self._predecessor_instances.update(dict(zip(self._parent_names, self._parent_instances)))

        self._is_frozen = True

    def __setattr__(self, key, value):
        if self._is_frozen and key in self._parent_names:
            raise AttributeError("Class {} parents are frozen: cannot set {} = {}".format(self.name, key, value))
        else:
            object.__setattr__(self, key, value)

    def get_parent_by_type(self, t):
        """Algorithm is breadth first."""
        if len(self._parent_instances) == 0:
            return None

        for par in self._parent_instances:
            if isinstance(par, t):
                return par

        for par in self._parent_instances:
            ret = par.get_parent_by_type(t)
            if ret is not None:
                return ret

        return None

    def get_parent_by_name(self, name):
        """Algorithm is breadth first."""
        if len(self._parent_instances) == 0:
            return None

        for par in self._parent_instances:
            if par.name == name:
                return par

        for par in self._parent_instances:
            ret = par.get_parent_by_name(name)
            if ret is not None:
                return ret

        return None

    def is_up_to_date(self):
        """Indicate wether this step should be processed again."""
        for par in self._parent_instances:
            if not par.is_up_to_date():
                return False

        return self.are_states_equal()

    def are_states_equal(self):
        """Return True is the state of the current object is the same as the one in the serialized json dump."""
        dict_json = self.load_state()

        if dict_json is None:
            return False

        try:
            for k in self.attributes_to_serialize:

                if str(dict_json[k]) != str(getattr(self, k)):
                    logger.debug("Key %s mismatch: left=%s / right=%s", k, str(dict_json[k]), str(getattr(self, k)))
                    return False

        except KeyError:
            return False

        except Exception, e:
            logger.warning("Exception caught in 'are_states_equal: %r", e)
            return False

        return True

    def serialize_state(self):
        """
        Flush the state of the runner into the json file mentioned by 'json_prefix' (init) to
        which the name of the current Job has been appended in the form 'json_prefix'_'name'.json
        Also flushes the state of the parents as well.
        """
        for par in self._parent_instances:
            par.serialize_state()

        # no need if there is no change in configuration
        if self.are_states_equal():
            return

        assert(self.json_filename is not None)

        d = {}

        # Paramters of the Job
        for k in self.attributes_to_serialize:
            d[k] = getattr(self, k)

        # Outputs of the Job
        for k in self.outputs_to_cache:
            d[k] = getattr(self, k)

        with open(self.json_filename, 'w') as f:
            json.dump(d, f)

    def load_state(self):
        """Load the json file."""
        if self.json_filename is None:
            return None

        if not os.path.exists(self.json_filename):
            logger.debug("Loading the state failed: File does not exist %s.", self.json_filename)
            return None

        dict_json = json.load(open(self.json_filename))

        return dict_json

    def run(self, *args, **kwargs):
        """
        Run this Job's computation/action. Should be overridden by an implementation class.

        :param *args: Outputs of the parent Jobs. The order of the args is determined by the
                      the order of self.parents.
        """
        raise RuntimeError("Should be overridden")

    def process(self):
        """
        Process the current node and all the parent nodes, and provide the outputs
        of the parents to this node.
        """
        if self.is_up_to_date():
            return

        # if not up to date, we need all the parents
        parent_outputs = []
        for par in self._parent_instances:
            par.process()
            parent_outputs.append(par.get_outputs())

        self.run(*parent_outputs)
        self.serialize_state()
        # after this call, the current instance should be up to date

    def is_output_cached(self):
        """Check if all output attributes are present."""
        for attr in self.outputs_to_cache:
            if getattr(self, attr, None) is None:
                return False

        return True

    def cache_output(self):
        """Set all output attributes as loaded from the JSON file."""
        json_state = self.load_state()

        if json_state is None:
            raise RuntimeError('Trying to cache output from non-existing JSON file.')

        for attribute in self.outputs_to_cache:
            setattr(self, attribute, json_state[attribute])

    def get_outputs(self):
        """Return all the possible outputs of this step."""
        if not self.is_up_to_date():
            raise RuntimeError("Cannot query for the outputs of Job {} before those are computed with the new parameters".format(self.name))

        if not self.is_output_cached():
            self.cache_output()

        return None
