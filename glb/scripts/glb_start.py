from argparse import ArgumentParser
import logging

from rich.console import Console
from rich.logging import RichHandler

import glb.core.gpu_master


log = logging.getLogger(__file__)


LOGGING_FORMAT = '%(message)s'
WELCOME_STR = """
   ____ _     ____         ____ ____  _   _           __  __           _            
  / ___| |   | __ )   _   / ___|  _ \| | | |         |  \/  | __ _ ___| |_ ___ _ __ 
 | |  _| |   |  _ \  (_) | |  _| |_) | | | |  _____  | |\/| |/ _` / __| __/ _ \ '__|
 | |_| | |___| |_) |  _  | |_| |  __/| |_| | |_____| | |  | | (_| \__ \ ||  __/ |   
  \____|_____|____/  (_)  \____|_|    \___/          |_|  |_|\__,_|___/\__\___|_|   
                                                                                    
"""
WELCOME_STR_STYLE = '#B87333'


def start():
    parser = ArgumentParser()
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    logging.basicConfig(
        level=(logging.DEBUG if args.debug else logging.INFO),
        format=LOGGING_FORMAT, 
        handlers=[RichHandler(
            locals_max_string=None, 
            tracebacks_word_wrap=False)])
    
    console = Console()
    console.print(WELCOME_STR, style=WELCOME_STR_STYLE)

    log.debug('DEBUG MODE')
    glb.core.gpu_master.serve(debug=args.debug)