import os
import sys
# import re
import signal
import functools
import inspect
import contextlib
import logging

import grpc

from glb.utils.utils import is_server_available
from glb.utils.gpu_monitors import SingleGPUMonitor
import glb.grpc_resources.master_pb2 as protos
import glb.grpc_resources.master_pb2_grpc as services
from glb.core.constants import GLB_SERVER_TMP_INFO_PATH, ServiceErrorCodes

from typing import (
    Callable, 
    Iterable, 
    Tuple, 
    Any,
)


log = logging.getLogger(__file__)


__all__ = ['job']


def __grpc_failed_job_handler(
    *args, 
    stub: services.GPUMasterStub, 
    jobtype: protos.JobType
) -> None:
    profile = protos.JobProfile(
        jobtype=jobtype,
        succeeded=False)
    stub.CompleteJob(profile, wait_for_ready=True)
    sys.exit()


@contextlib.contextmanager
def __grpc_handle_signals(
    sigs: Iterable[signal.Signals],
    stub: services.GPUMasterStub, 
    jobtype: protos.JobType, 
) -> None:
    orig_handlers = [signal.getsignal(s) for s in sigs]
    for s in sigs:
        signal.signal(s, functools.partial(
            __grpc_failed_job_handler, 
            stub=stub, jobtype=jobtype))
    yield
    # Restore control to original handlers
    for s, handler in zip(sigs, orig_handlers):
        signal.signal(s, handler)


def __request_gpu(
    stub: services.GPUMasterStub, 
    jobtype: protos.JobType
) -> int:
    response: protos.GPU = stub.RequestGPU(jobtype, wait_for_ready=True)
    errorcode = ServiceErrorCodes(response.errorcode)
    if errorcode == ServiceErrorCodes.EXCEEDS_TOTAL_MEMORY:
        raise MemoryError(f'{errorcode}: Cannot complete job \n```\n{jobtype}```\n')
    return response.gpu_id


# def __parse_error_for_job_info(error: str) -> Optional[float]:
#     # TODO: This needs to be documented better / made more robust.
#     # String search for the amount of memory of the attempted allocation.
#     # This memory reported by CUDA is in GiB, so we have to convert it to
#     # MB before sending it.
#     if re.search(r'out of memory', error):
#         alloc = re.search(r'(Tried to allocate )(\d+.\d+)', error)
#         if alloc is None:
#             mem_used = float('inf')
#         else:
#             mem_used = float( alloc.group(2) )
#             mem_used = int(mem_used) * 8e6  # convert to bits
#         return mem_used


def __run_job(
    job_func: Callable, 
    gpu_id: int, 
    jobtype: protos.JobType
) -> Tuple[Any, protos.JobProfile]:
    os.environ['CUDA_VISIBLE_DEVICES'] = f'{gpu_id}'

    # Collects job profile stats in separate thread.
    with SingleGPUMonitor(gpu_id, delay=0.01) as monitor: 
        mem_used = None
        try:
            outputs = job_func()
        except RuntimeError as e:
            # mem_used = __parse_error_for_job_info(str(e))
            # if mem_used is None:
            #     raise e
            print(e)
            outputs = None
            profile = protos.JobProfile(
                jobtype=jobtype,
                succeeded=False)
        else:
            if mem_used is None:
                mem_used = monitor.max_mem_used
            else:
                mem_used = max(monitor.max_mem_used, mem_used)

            profile = protos.JobProfile(
                jobtype=jobtype,
                succeeded=True,
                gpu=protos.GPU(
                    gpu_id=gpu_id, 
                    errorcode=0),
                max_gpu_memory_used=mem_used,
                max_gpu_load=monitor.max_load)
    
    return outputs, profile


def __find_gpu_master_address() -> str:
    # Find the GPU Master port
    if os.path.isfile(GLB_SERVER_TMP_INFO_PATH):
        with open(GLB_SERVER_TMP_INFO_PATH, 'r') as glb_file:
            gpu_master_addr = f'localhost:{glb_file.read()}'
    # File containing current server port must exist.
    else:
        raise FileNotFoundError(f'{GLB_SERVER_TMP_INFO_PATH}, serve never called.')
    return gpu_master_addr


def job(
    continue_on_server_unavailable: bool = False,
) -> Callable:

    def load_balance(
        func: Callable,
    ) -> Callable:
        # Construct the jobstr that specifies the job type.
        func_file = inspect.getfile(func)
        __JOBSTR = (
            f'{func_file}::{func.__class__.__name__}' if inspect.isclass(func)
            else f'{func_file}::{func.__name__}')
        __JOBTYPE = protos.JobType(jobstr=__JOBSTR)

        __REGISTRY = {}
        # This function is added as an attribute of the decorator at the end.
        # This allows it to be invoked as a decorator itself.
        def __profile(pfunc: Callable) -> Callable:
            if __REGISTRY.get('profiler') is not None:
                raise AttributeError(
                    f'Tried to register profiler {inspect.getfile(pfunc)}::{pfunc.__name__} '
                    f'for {__JOBSTR}, but one already exists: '
                    f'{inspect.getfile(__REGISTRY["profiler"])}::{__REGISTRY["profiler"].__name__}')
            __REGISTRY['profiler'] = pfunc
            return pfunc
 
        @functools.wraps(func)
        def decorator(*args, **kwargs):
            try:
                __GPU_MASTER_ADDR = __find_gpu_master_address()
            except FileNotFoundError as e:
                if not continue_on_server_unavailable:
                    raise e
                __GPU_MASTER_ADDR = 'None'

            # If the server is unavailable, then raise exception by default.
            # Otherwise, proceed with computation without load balancing.
            if not is_server_available(__GPU_MASTER_ADDR):
                if not continue_on_server_unavailable:
                    raise ConnectionError('GPU Master is not available.')
                log.warn(
                    'GPU Master is not running but continue_on_server_unavailable=True, '
                    'proceeding anyway... This may result in unexpected job failures.')
                return func(*args, **kwargs)

            with grpc.insecure_channel(
                __GPU_MASTER_ADDR, 
                options=(('grpc.enable_http_proxy', 0),)
            ) as channel:
                stub = services.GPUMasterStub(channel)

                # We need to communicate abnormal process termination to the server.
                # TODO: Using a gRPC stream should eliminate the need for this signal
                #       handling.
                with __grpc_handle_signals(
                    (signal.SIGINT, signal.SIGTERM), stub, __JOBTYPE
                ):
                    # Run the profile job if this is the first invocation of this job
                    # relative to the server lifetime.
                    # If no profiler was registered, then run the full job as the profile
                    # job and return its outputs.
                    is_job_profiled = stub.JobTypeExists(__JOBTYPE).response
                    if not is_job_profiled: 
                        job_func = __REGISTRY.get(
                            (lambda: __REGISTRY['profiler'](*args, **kwargs)), 
                            (lambda: func(*args, **kwargs)))
                        gpu_id = __request_gpu(stub, __JOBTYPE)
                        outputs, profile = __run_job(
                            job_func=job_func,
                            gpu_id=gpu_id, 
                            jobtype=__JOBTYPE)
                        stub.CompleteJob(profile)

                    # If a profiler was registered, now run the actual job and 
                    # return its outputs. If the profile failed, don't run the full
                    # job.
                    # The other case is that this job type has already been profiled,
                    # which means we should just run it.
                    if (
                        is_job_profiled or (
                            (not is_job_profiled) and 
                            (profile.succeeded) and 
                            (__REGISTRY.get('profiler') is not None))
                    ):
                        gpu_id = __request_gpu(stub, __JOBTYPE)
                        outputs, profile = __run_job(
                            job_func=(lambda: func(*args, **kwargs)),
                            gpu_id=gpu_id,
                            jobtype=__JOBTYPE)
                        stub.CompleteJob(profile)

            return outputs

        decorator.profile = __profile
        return decorator
    return load_balance
