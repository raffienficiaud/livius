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
                    # type=str,
                    # nargs='+',
                    help='folder containing videos')
parser.add_argument('--list-workflows',
                    action='store_true',
                    help='lists all available workflows and exits')
parser.add_argument('--workflow',
                    dest='workflow',
                    # action='store_const',
                    help='specifies the workflow to use')
parser.add_argument('--temporary-folder',
                    dest='temp_folder',
                    # action='store_const',
                    help='specifies the workflow to use')

args = parser.parse_args()

# list workflow
if(args.list_workflows):
    import video.processing.workflow as WORK
    import inspect

    prepend = '\t'

    for entry in dir(WORK):
        entry_obj = getattr(WORK, entry)
        if(inspect.isfunction(entry_obj) and entry.find('workflow') == 0):
            print
            print '#' * 15
            print 'workflow:', entry
            print prepend + inspect.getdoc(entry_obj).replace('\n', '\n' + prepend)


    sys.exit(0)


