

"""This file contains the "preparation" functions from the video, such as the

  - creation of the thumnail images
  - computation of the min/max/histograms of the appropriate regions in frames

"""

import os
import json
import logging
from exceptions import AttributeError

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
logger.debug('')

class Job(object):

    name = "root"
    attributes_to_serialize = []

    parents = None

    # private API
    _is_frozen = False

    @classmethod
    def add_parent(cls, obj):
        """Add a specific job as a parent job.

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
        """Returns all the parents jobs of this class"""
        return cls.parents

    def __init__(self, *args, **kwargs):
        """
        :param json_prefix: the prefix used for serializing the state of this runner
        """
        super(Job, self).__init__()

        self._is_frozen = False

        json_prefix = kwargs.get('json_prefix', '')
        self.json_filename = json_prefix + '_' + self.name + '.json'

        # creation of the serialized attributes
        for k in self.attributes_to_serialize:
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
        """Algorithm is breadth first"""
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
        """Algorithm is breadth first"""
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
        """Contains the logic to indicate that this step should be processed again"""

        for par in self._parent_instances:
            if not par.is_up_to_date():
                return False

        return self.are_states_equal()

    def are_states_equal(self):
        """Returns True is the state of the current object is the same as the one in the serialized json dump"""

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
        """Flushes the state of the runner into the json file mentioned by 'json_prefix' (init) to
        which the name of the current Job has been appended in the form 'json_prefix'_'name'.json
        Also flushes the state of the parents as well
        """

        for par in self._parent_instances:
            par.serialize_state()

        # no need if there is no change in configuration
        if self.are_states_equal():
            return

        assert(self.json_filename is not None)

        d = {}
        for k in self.attributes_to_serialize:
            d[k] = getattr(self, k)

        with open(self.json_filename, 'w') as f:
            json.dump(d, f)

    def load_state(self):
        """
        Loads the json file.
        """
        if self.json_filename is None:
            return None

        if not os.path.exists(self.json_filename):
            logger.debug("Loading the state failed: File does not exist %s.", self.json_filename)
            return None

        dict_json = json.load(open(self.json_filename))

        return dict_json

    def run(self, *args, **kwargs):
        """Runs this specific action: should be overridden by an implementation class."""
        raise RuntimeError("Should be overridden")

    def process(self):
        """Process the current node and all the parent nodes, and provides the outputs
        of the parents to this node."""

        if self.is_up_to_date():
            return

        # if not up to date, we need all the parents
        parent_outputs = []
        for par in self._parent_instances:
            par.process()
            parent_outputs.append(par.get_outputs())

        self.run(*parent_outputs)

        # after this call, the current instance should be up to date

    def get_outputs(self):
        """Returns all the possible outputs of this step"""
        if not self.is_up_to_date():
            raise RuntimeError("Cannot query for the outputs of Job {} before those are computed".format(self.name))

        return None
