"""
Metadata
========

This file contains a Job for extracting the metadata from a user annotated file. The format is specific to the
MLSS 2015, but may be easily adapted to your needs.

.. autosummary::

  Metadata
  metadata_mogrifier

"""


from ..job import Job

import os
import logging

logger = logging.getLogger()


class Metadata(Job):
    """
    Job taking the metadata location anf video filename and produce metadata suitable
    for the movie composition object :py:class:`ClipsToMovie <livius.video.processing.jobs.create_movie.ClipsToMovie>`.

    The expected structure of the meta data is the following:

    .. code::

        root of metadata
        |- video_file_name_WE
           |- video_file_name_WE_metadata_input.json
           |- some_png_image.png


    where ``WE`` stands for *without extension*. The meta will look for the files containing the relevant
    information under the root directory containing the meta data, and the video filename WE. It will look for
    on picture in PNG format and consider it as the introduction picture.

    .. rubric:: Runtime parameters

    * ``video_filename`` name of the video to process (mandatory, cached)
    * ``meta_location`` location of the meta data (mandatory, not cached)

    .. note::

      Some of the attributes of the final video are set by the
      :py:class:`ClipsToMovie <livius.video.processing.jobs.create_movie.ClipsToMovie>`
      job (such as the credit/epilog
      images, the background image, etc).

    .. rubric:: Metadata file format

    As its name indicates the file ``video_file_name_WE_metadata_input.json`` is a json file containing the following fields:

    * ``speaker`` the name of the speaker
    * ``title`` the title of the talk
    * ``date`` in a string format (it will not be interpreted in any way)
    * ``video_begin`` (optional) indicates the beginning of the sequence in a moviePy format (eg. 00:00:57 for 57sec)
    * ``video_end`` (optional) indicates the end of the sequence in a moviePy format

    """

    name = "movie_metadata"

    #: Cached input:
    #:
    #: * ``video_filename`` input video file
    #: * ``intro_image_names`` the name of the image shown in introduction
    #: * ``video_begin`` beginning of the video
    #: * ``video_end`` end of the video
    #: * ``title`` title of the talk
    #: * ``speaker`` speaker of the talk
    #: * ``date`` date of the talk (full string)
    #:
    #: All the parameters read from the metadata file are considered as inputs to the Job. If we do otherwise
    #: (eg. cached outputs), then it might happen that the job is not properly flagged as dirty when some
    #: settings change (eg. adding an image file).
    attributes_to_serialize = ['video_filename',
                               'intro_image_names',
                               'video_begin',
                               'video_end',
                               'title',
                               'speaker',
                               'date']

    #: Cached output:
    #:
    outputs_to_cache = []

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

    def _get_meta_location(self):
        return os.path.join(self.meta_location,
                            os.path.splitext(self.video_filename)[0])

    def _get_meta_filename(self):
        return os.path.join(self._get_meta_location(),
                            os.path.splitext(self.video_filename)[0] + '_metadata_input.json')

    class __dummy(object):
        pass

    def _read_file_content(self):
        with open(self._get_meta_filename()) as f:
            import json
            d = json.load(f)

            self.title = d['title']
            self.speaker = d['speaker']
            self.date = d['date']
            self.video_begin = d['video_begin'] if 'video_begin' in d else ""  # empty string evaluates to False and is serializable
            self.video_end = d['video_end'] if 'video_end' in d else ""

        # listing the image files, sorting them, and considering them as introduction images.
        l = os.listdir(os.path.dirname(self._get_meta_filename()))
        self.intro_image_names = [i for i in l if os.path.splitext(i)[1].lower() == '.png']
        self.intro_image_names.sort()

    def run(self, *args, **kwargs):
        # doing nothing in particular
        pass

    def get_outputs(self):
        super(Metadata, self).get_outputs()
        logger.info('[METADATA] information extracted for video %s', self.video_filename)

        return {'talk_title': self.title,
                'speaker_name': self.speaker,
                'talk_date': self.date,
                'intro_images': [os.path.join(self._get_meta_location(), i) for i in self.intro_image_names],
                'video_begin': self.video_begin if self.video_begin != "" else None,
                'video_end': self.video_end if self.video_end != "" else None
                }


def metadata_mogrifier(folder):
    """Utility function transforming metadata stored in txt file into a format usable by
    :py:class:`Metadata <livius.video.processing.jobs.meta.Metadata>` usable one.

    This is just an example reading 2 files in a JSON format and creating the appropriate metadata input
    for the :py:class:`Metadata <livius.video.processing.jobs.meta.Metadata>` class.

    It is possible to run directly this function by providing the location of the folder to parse, in the following way::

      python -m livius.video.processing.jobs.meta some_folder

    """
    import json

    list_videos = os.listdir(folder)

    for current_video in list_videos:
        dout = {}
        full_path = os.path.abspath(os.path.join(folder, current_video))

        video_filename = current_video

        metadata_1 = os.path.join(full_path, video_filename + '.txt')

        # this one should exist: main metadata
        assert(os.path.exists(metadata_1))

        if os.path.exists(metadata_1):
            with open(metadata_1) as f:
                d = json.load(f)['TalkDetail']

                dout['title'] = d[1]
                dout['speaker'] = d[0]
                dout['date'] = d[3]

        segment_file = os.path.join(full_path, 'processing_%s_Time_Total_Seconds.json' % video_filename)
        if os.path.exists(segment_file):
            with open(segment_file) as f:
                d = json.load(f)['CuttingTimes']
                if(d[0].lower() == 'yes'):
                    dout['video_begin'] = d[1]
                    dout['video_end'] = d[2]

        out_file = os.path.join(full_path, video_filename + '_metadata_input.json')
        with open(out_file, 'w') as f:
            json.dump(dout, f, indent=4)

if __name__ == '__main__':
    import sys
    print sys.argv
    metadata_mogrifier(sys.argv[1])
