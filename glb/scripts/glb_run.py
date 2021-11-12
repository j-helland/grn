"""
Start a GLB GPU Master, run the command, then terminate the GPU Master.
"""
import time
from argparse import ArgumentParser
import subprocess
import logging

from rich.logging import RichHandler


log = logging.getLogger(__file__)
LOGGING_FORMAT = '%(message)s'


def run():
    parser = ArgumentParser()
    parser.add_argument('cmds', type=str, nargs='+')
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.WARNING,
        format=LOGGING_FORMAT, 
        handlers=[RichHandler(
            locals_max_string=None, 
            tracebacks_word_wrap=False)])

    proc = subprocess.Popen(['glb-start'])
    time.sleep(0.5)
    procs = [subprocess.Popen(cmd.split(' ')) for cmd in args.cmds]
    for p in procs:
        p.wait()
    proc.terminate()
