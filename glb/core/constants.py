import pathlib
from enum import Enum, auto

__all__ = ['GLB_SERVER_TMP_INFO_PATH', 'ServiceErrorCodes']


# Obtain a universally consistent location to cache intermediate info needed by clients.
GLB_SERVER_TMP_INFO_PATH = str( pathlib.Path(__file__).resolve().parent / '.glb.conf' )

class ServiceErrorCodes(Enum):
    OK = auto()
    EXCEEDS_TOTAL_MEMORY = auto()
    EXCEEDS_CURRENT_MEMORY = auto()
    WAITING_FOR_JOB_PROFILE = auto()