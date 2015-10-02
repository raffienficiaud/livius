"""
Metadata
========

This file contains a Job for extracting the metadata from a user annotated file. The format is specific to the
MLSS 2015, but may be easily adapted to your needs.

"""


from ..job import Job

import datetime
import os
import logging


logger = logging.getLogger()

class Metadata(Job):
    """
    Job taking the metadata location anf video filename and produce metadata suitable
    for the movie composition object :py:class:`ClipsToMovie`.

    .. rubric:: Runtime parameters

    * ``video_filename`` name of the video to process
    *

    """

    name = "movie_metadata"

    #: Cached input:
    #:
    #: * ``video_filename`` input video file
    attributes_to_serialize = ['video_filename']

    #: Cached output:
    #:
    #: * ``intro_image_names`` the name of the image shown in introduction
    #: * ``video_begin`` beginning of the video
    #: * ``video_end`` end of the video
    outputs_to_cache = ['intro_image_names',
                        'video_begin',
                        'video_end',
                        'title',
                        'speaker',
                        'date'
                        ]

    def __init__(self, *args, **kwargs):
        """
        :param str video_filename: video being processed
        """

        super(Metadata, self).__init__(*args, **kwargs)

        assert('video_filename' in kwargs)

        # unicode is for comparison issues when retrieved from the json
        self.video_filename = unicode(self.video_filename)

    def run(self, *args, **kwargs):

        self.title = "test"
        self.speaker = "me"
        self.date = 'now'
        pass

    def get_outputs(self):
        super(Metadata).get_outputs()
        logger.info('[METADATA] processed metadata for video %s', self.video_filename)

        return {'talk_title': self.title,
                'talk_speaker': self.speaker,
                'talk_date': self.date
                }
