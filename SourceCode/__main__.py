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
parser.add_argument('--thumbnails-folder',
                    help='specifies the folder where the thumbnails will be stored/retrieved')
parser.add_argument('--output-folder',
                    help='specifies the output folder')
parser.add_argument('--process-only-index',
                    metavar='INDEX',
                    help='''process only the file specified by the INDEX. The files are sorted
                    so that this option may be used as an option for dispatching the processing
                    of all the files on different machines (such as a cluster)''')
parser.add_argument('--option',
                    metavar='KEY=VALUE',
                    help="""Additional runtime option. Each parameter has the form --option=key=value.
                    This option may appear multiple times. """,
                    action='append')
parser.add_argument('--non-interactive',
                    help='indicates that the processing will be not interactive (mainly for matplotlib backend)',
                    action='store_true')
parser.add_argument('--option-file',
                    metavar='FILE.json',
                    help="""Reads a set of additional runtime options from a json file.
                    This options in this file may be overriden by the --option.""")

parser.add_argument('--print-workflow',
                    action='store_true',
                    help="""Prints the workflow (text) and exits.""")
parser.add_argument('--dot-workflow',
                    action='store_true',
                    help="""Prints the workflow (dot) and exits.""")

parser.add_argument('--is-visual-test',
                    action='store_true',
                    help="""If set on the command line, the video is processed only for 10 seconds.
                    This however does not prevent the full thumbnail extraction.""")


args = parser.parse_args()

# getting loggers
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

# this should be the first thing
if args.non_interactive is not None and args.non_interactive:
    import matplotlib
    matplotlib.use('Agg')


# list workflow
if(args.list_workflows):
    import video.processing.workflow as workflow_module_inspect
    import inspect

    prepend = '\t'

    for entry in dir(workflow_module_inspect):
        entry_obj = getattr(workflow_module_inspect, entry)
        if(inspect.isfunction(entry_obj) and entry.find('workflow') == 0):
            print
            print '#' * 15
            print 'workflow:', entry
            print prepend + inspect.getdoc(entry_obj).replace('\n', '\n' + prepend)

    sys.exit(0)

if not args.workflow:
    logger.error("[CONFIG] a workflow should be specified")
    sys.exit(1)

# loads the workflow
try:
    import video.processing.workflow as workflow_module
    workflow_factory_obj = getattr(workflow_module, args.workflow)
except Exception, e:
    logger.error('[CONFIG] the workflow %s cannot be loaded (Error: %s)', args.workflow, e)
    sys.exit(1)

if args.print_workflow or args.dot_workflow:
    if args.print_workflow:
        print workflow_factory_obj().workflow_to_string()
    else:
        print workflow_factory_obj().workflow_to_dot()
    sys.exit(0)

if not args.video_folder and not args.video_file:
    logger.error("[CONFIG] one or more video files or folders should be specified")
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

# loads the video files
video_files = args.video_file if args.video_file is not None else []
if args.video_folder is not None:
    for f in args.video_folder:
        video_files += [os.path.abspath(os.path.join(f, i)) for i in os.listdir(f)]

if not video_files:
    logger.error('[CONFIG] the video list to be processed is empty')
    sys.exit(1)

if not args.output_folder:
    logger.error('[CONFIG] the video list to be processed is empty')
    sys.exit(1)

args.output_folder = os.path.abspath(args.output_folder)

if not args.thumbnails_folder:
    args.thumbnails_folder = os.path.join(args.output_folder, 'thumbnails')

logger.info("[CONFIG] output folder %s", args.output_folder)
logger.info("[CONFIG] thumbnails folder %s", args.thumbnails_folder)
logger.info("[CONFIG] workflow %s", args.workflow)

# sorting the videos so that they do not depend on the order given by the file
# system
video_files.sort()
for e, f in enumerate(video_files):
    s = ''
    if args.process_only_index is not None:
        if e == int(args.process_only_index):
            s = '  ** processing **'
        else:
            s = '  (not processing)'
    logger.info("[VIDEO] -> %s <-%s", f, s)


options = {}
if args.option_file:
    if not os.path.exists(args.option_file):
        logger.error('[CONFIG] the option file %s does not exist', args.option_file)
        sys.exit(1)

    with open(args.option_file) as f:
        import json
        d = json.load(f)
        options.update(d)

if args.option:
    for v in args.option:
        s = v.split('=')
        if s is not None:
            options[s[0]] = s[1]

# we need only one instance of the workflow as the parents are static fields
workflow_instance = workflow_factory_obj()

# process all files
for index, f in enumerate(video_files):

    if args.process_only_index is not None:
        if index != int(args.process_only_index):
            continue

    video_base_name = os.path.splitext(os.path.basename(f))[0]
    output_location = os.path.join(args.output_folder, video_base_name)
    if not os.path.exists(output_location):
        os.makedirs(output_location)

    thumbnails_root = os.path.join(args.thumbnails_folder, video_base_name)
    if not os.path.exists(thumbnails_root):
        os.makedirs(thumbnails_root)

    params = options.copy()

    # those important parameter should not be overriden
    params.update({'video_filename': os.path.basename(f),
                   'video_location': os.path.dirname(f),
                   'thumbnails_root': thumbnails_root,
                   'json_prefix': os.path.join(output_location, video_base_name),
                   'is_visual_test': args.is_visual_test is not None and args.is_visual_test
                   })

    outputs = workflow_module.process(workflow_instance, **params)
    # outputs.write_videofile(os.path.join(slide_clip_folder, 'slideclip.mp4'))
