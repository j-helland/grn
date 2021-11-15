import depq

from threading import Lock, Event
from typing import (
    Dict, 
    Union, 
    Optional,
)
NumericType = Union[int, float]


__all__ = ['GPUStates', 'JobStates']


class GPUStates:
    STATES: Dict[int, Dict[str, NumericType]] = dict()
    STATES_INIT = Event()
    LOCK = Lock()
    GPU_QUEUE: depq.DEPQ = depq.DEPQ()
    PROFILING_GROUP: Dict[int, str] = dict()

    # TODO: This should probably be defined in constants.py
    TARGET_RESOURCE: str = 'memoryUsed'


class JobStates:
    JOB_TYPES = dict()
    ACTIVE_JOBS = dict()
    MAX_NUM_JOBS: Optional[int] = None
    LOCK = Lock()