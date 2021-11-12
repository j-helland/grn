from threading import Lock, Event
from typing import Dict, List, Tuple, Union, Optional
NumericType = Union[int, float]


__all__ = ['GPUStates', 'JobStates']


class GPUStates:
    STATES: Dict[int, Dict[str, NumericType]] = dict()
    STATES_INIT = Event()
    LOCK = Lock()
    HEAP: List[Tuple[float, int]] = []
    PROFILING_GROUP: Dict[int, str] = dict()

    # TODO: This should probably be defined in constants.py
    RESOURCE_POLICY: str = 'memoryUsed'


class JobStates:
    JOB_TYPES = dict()
    ACTIVE_JOBS = dict()
    MAX_NUM_JOBS: Optional[int] = None
    LOCK = Lock()
    READY = Event()