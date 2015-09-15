"""
Main entry point of the Package
"""


import argparse
import sys
import os
import logging

# Logging facility, just in case
# the 7s should contain the WARNING message
logging.basicConfig(format='%(asctime)s | %(levelname)-7s | %(message)s', level=logging.DEBUG)


# description
description = """Livius video processing.

This project automate the process of creating
video files from recorded presentation. It takes asa input
a directory containing videos (or a single video)
"""

sys.argv[0] = os.path.dirname(__file__)

#
parser = argparse.ArgumentParser(description=description,
                                 usage='python -m %(prog)s [options]')
parser.add_argument('--video-folder',
                    metavar='VDIR',
                    help="""Indicates a folder containing videos to be processed. All videos in the specified folders
                    will be processed (subfolders are not explored). This option
                    may appear multiple times.""",
                    action='append')
parser.add_argument('--video-file',
                    metavar='VFILE',
                    help='Specific video file to process. This option may appear multiple time.',
                    action='append')
parser.add_argument('--list-workflows',
                    action='store_true',
                    help='lists all available workflows and exits')
parser.add_argument('--workflow',
                    help='specifies the workflow to use')
parser.add_argument('--temporary-folder',
                    help='specifies the workflow to use')

args = parser.parse_args()

# getting loggers
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

# list workflow
if(args.list_workflows):
    import video.processing.workflow as workflow_module
    import inspect

    prepend = '\t'

    for entry in dir(workflow_module):
        entry_obj = getattr(workflow_module, entry)
        if(inspect.isfunction(entry_obj) and entry.find('workflow') == 0):
            print
            print '#' * 15
            print 'workflow:', entry
            print prepend + inspect.getdoc(entry_obj).replace('\n', '\n' + prepend)


    sys.exit(0)

if not args.video_folder and not args.video_file:
    logger.error("[CONFIG] one or more video files or folders should be specified")
    sys.exit(1)

if not args.workflow:
    logger.error("[CONFIG] a workflow should be specified")
    sys.exit(1)

if args.video_folder:
    for f in args.video_folder:
        if not os.path.exists(f):
            logger.error("[CONFIG] the specified video folder does not exist: %s", f)
            sys.exit(1)

if args.video_file:
    for f in args.video_file:
        if not os.path.exists(f):
            logger.error("[CONFIG] the specified video file does not exist: %s", f)
            sys.exit(1)

# loads the workflow
import video.processing.workflow as workflow_module
try:
    workflow_factory_obj = getattr(workflow_module, args.workflow)
except Exception, e:
    logger.error('[CONFIG] the workflow %s cannot be loaded', args.workflow)
    sys.exit(1)

# loads the video files
video_files = args.video_file if args.video_file is not None else []
if args.video_folder is not None:
    for f in args.video_folder:
        video_files += os.listdir(f)

if not video_files:
    logger.error('[CONFIG] the video list to be processed is empty')
    sys.exit(1)

for f in video_files:
    logger.info("[FILE] -> %s <- will be processed", f)

if not args.temporary_folder:
    args.temporary_folder = os.path.abspath(os.path.join(os.path.dirname(video_files[0]), 'tmp_livius'))

# process all files
for f in video_files:
    params = {'video_filename': f,
              'thumbnails_location': os.path.join(args.temporary_folder, os.path.basename(f), 'thumbnails'),
              # 'json_prefix': os.path.join(proc_folder, 'processing_video_7_'),
              # 'segment_computation_tolerance': 0.05,
              # 'segment_computation_min_length_in_seconds': 2,
              # 'slide_clip_desired_format': [1280, 960],
              # 'nb_vertical_stripes': 10
              }

    outputs = workflow_module.process(workflow_factory_obj(), **params)
    # outputs.write_videofile(os.path.join(slide_clip_folder, 'slideclip.mp4'))

