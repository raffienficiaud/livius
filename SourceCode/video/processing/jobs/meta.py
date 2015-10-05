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

    The expected structure of the meta data is the following:

    root of metadata
    |- ``video_file_name_WE``
    |-- ``video_file_name_WE.txt``
    |-- ``some_png_image.png``

    where ``WE`` stands for /without extension/. The meta will look for the files containing the relevant
    information under the root directory containing the meta data, and the video filename WE. It will look for
    on picture in PNG format and consider it as the introduction picture.

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
        :param str video_filename: video being processed (cached)
        :param str meta_location: location of the meta information (not cached)
        """

        super(Metadata, self).__init__(*args, **kwargs)

        assert('video_filename' in kwargs)
        assert('meta_location' in kwargs)

        # unicode is for comparison issues when retrieved from the json
        self.video_filename = unicode(self.video_filename)

        if not os.path.exists(self._get_meta_filename()):
            logger.error("""[META] the file %s was not found. Please ensure the meta information
                         of the videos have the proper layout""", self._get_meta_filename())
            raise RuntimeError("[META] file %s not found" % self._get_meta_filename())

        self._read_file_content()

    def _get_meta_filename(self):
        return os.path.join(self.meta_location,
                            os.path.splitext(self.video_filename)[0],
                            os.path.splitext(self.video_filename)[0] + '.txt')

    def _read_file_content(self):
        with open(self._get_meta_filename()) as f:
            import json
            d = json.load(f)['TalkDetail']

            self.title = d[1]
            self.speaker = d[0]
            self.date = d[3]

        l = os.listdir(os.path.dirname(self._get_meta_filename()))
        self.intro_image_names = [i for i in l if os.path.splitext(i)[1].lower() == '.png']

    def is_up_to_date(self):
        """is_up_to_date returns False if there was any update of the meta files"""

        if not os.path.exists(self.json_filename):
            return False

        if(os.stat(self._get_meta_filename()).st_mtime >= os.stat(self.json_filename).st_mtime):
            return False

        return super(Metadata, self).is_up_to_date()

    def run(self, *args, **kwargs):
        pass

    def get_outputs(self):
        super(Metadata, self).get_outputs()
        logger.info('[METADATA] information extracted for video %s', self.video_filename)

        return {'talk_title': self.title,
                'talk_speaker': self.speaker,
                'talk_date': self.date,
                'intro_images': self.intro_image_names
                }
