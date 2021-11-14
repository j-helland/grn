import pathlib
import tempfile
from enum import Enum, auto

__all__ = ['GLB_SERVER_TMP_INFO_PATH', 'ServiceErrorCode']


# Obtain a universally consistent location to cache intermediate info needed by clients.
GLB_SERVER_TMP_INFO_PATH = pathlib.Path(tempfile.gettempdir()) / '.glb.conf'

class ResourcePolicy(Enum):
    SPREAD = auto()
    PACK = auto()

class ServiceErrorCode(Enum):
    OK = auto()
    EXCEEDS_TOTAL_MEMORY = auto()
    EXCEEDS_CURRENT_MEMORY = auto()
    WAITING_FOR_JOB_PROFILE = auto()