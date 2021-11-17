"""
TODO:
[ ] Implement segmented heap to handle heterogeneous hardware resources.

TODO (BACKLOG):
[ ] Handle many connections. 
    Probably want to put the server into some kind of not ready state.
    Can possibly implement this behavior using gRPC service intercept handler.
[ ] NEED to write tests to identify bugs in my shitty multithreading code.
[ ] Graceful handling of endlessly spinning jobs. Possibly implement a timeout?
[ ] Continuous job profiling might be useful as a way to gracefully handle interruptions / failures.
    That way, there doesn't need to be special logic in the client to send a message when abnormal
    termination happens.
    Implement this using a gRPC stream.
"""
import sys
import os
# import time
from concurrent import futures
# import threading
# from threading import Thread
import signal
import logging

import grpc

from grn.utils.utils import find_free_port
from grn.utils.gpu_monitors import GPUMonitor
import grn.grpc_resources.master_pb2 as protos
import grn.grpc_resources.master_pb2_grpc as services
from grn.core.constants import GRN_SERVER_TMP_INFO_PATH, ServiceErrorCode, ResourcePolicy
from grn.core.globals import GPUStates, JobStates

from typing import Optional, Callable


log = logging.getLogger(__file__)


__all__ = ['serve']


# class BackgroundEventTrigger(Thread):
#     """
#     """
#     def __init__(self, event: threading.Event, delay=5):
#         super().__init__()
#         self.event = event
#         self.stopped = False
#         self.delay = delay
#         self.start()

#     def run(self):
#         while not self.stopped:
#             time.sleep(self.delay)
#             self.event.set()

#     def __enter__(self):
#         return self

#     def __exit__(self, type, value, traceback):
#         self.stopped = True


def get_next_available_gpu(jobstr: str, resource_policy: ResourcePolicy) -> protos.GPU:
    GPUStates.STATES_INIT.wait()
    is_new_job_type = (JobStates.JOB_TYPES.get(jobstr) is None)

    if len(GPUStates.GPU_QUEUE) == 0:
        errorcode = ServiceErrorCode.WAITING_FOR_JOB_PROFILE

    else:
        with GPUStates.LOCK:
            # FIXME: We assume GPU homogeneity -- all GPUs have identical architecture.
            # TODO: Implement a segmented heap to group homogenous hardware together.
            #       Then we can have a resource heap for each bin.

            def _find_first_non_profiling_gpu(pop_func: Callable) -> int:
                num_gpus = len(GPUStates.GPU_QUEUE)
                i = 0
                gid = None
                while (i < num_gpus) and len(GPUStates.GPU_QUEUE):
                    gid = pop_func()[0]
                    if gid not in GPUStates.PROFILING_GROUP:
                        break
                    i += 1
                else:
                    return None
                return gid


            if is_new_job_type:
                gid = _find_first_non_profiling_gpu(GPUStates.GPU_QUEUE.poplast)
            else:
                if resource_policy == ResourcePolicy.SPREAD:
                    gid = _find_first_non_profiling_gpu(GPUStates.GPU_QUEUE.poplast)
                elif resource_policy == ResourcePolicy.PACK:
                    gid = _find_first_non_profiling_gpu(GPUStates.GPU_QUEUE.popfirst)
                else:
                    raise NotImplementedError(resource_policy)


            def _handle_pack_policy(gid: int, mem_request: int) -> Optional[int]:
                i = 0
                group = [ (gid, GPUStates.STATES[gid]['memoryUsed']) ]
                while (i < len(GPUStates.GPU_QUEUE)):
                    if (GPUStates.STATES[group[-1][0]]['memoryFree'] >= mem_request):
                        break
                    group.append( GPUStates.GPU_QUEUE.popfirst() )

                else:
                    GPUStates.GPU_QUEUE.extend(group)
                    return None

                # Reinsert all but last element; last element will be given new resource priority.
                GPUStates.GPU_QUEUE.extend(group[:-1])  
                return group[-1][0]
            

            def _handle_spread_policy(gid: int, mem_request: int) -> Optional[int]:
                if GPUStates.STATES[gid]['memoryFree'] < mem_request:
                    return None
                return gid


            # If we reached the end of the heap, there must be no devices that aren't locked.
            if gid is None:
                errorcode = ServiceErrorCode.WAITING_FOR_JOB_PROFILE
            
            else:
                mem_free = GPUStates.STATES[gid]['memoryFree']
                mem_total = GPUStates.STATES[gid]['memoryTotal']
                # NOTE: Single python dict op should be inherently thread-safe.
                requested_memory = JobStates.JOB_TYPES.get(jobstr)

                log.debug(f'[get_next_available_gpu] (total) {mem_total}, (free) {mem_free}, (request) {requested_memory}')

                errorcode = ServiceErrorCode.OK
                if requested_memory is not None:
                    if resource_policy == ResourcePolicy.SPREAD:
                        gid = _handle_spread_policy(gid, requested_memory)
                    elif resource_policy == ResourcePolicy.PACK:
                        gid = _handle_pack_policy(gid, requested_memory)
                    else:
                        raise NotImplementedError(resource_policy)
                    
                    if gid is None:
                        errorcode = ServiceErrorCode.EXCEEDS_CURRENT_MEMORY
                    else:
                        # Reinsert gpu with new resource priority.
                        # We need to update GPU state right now so that other threads
                        # will immediately become aware of new resource constraints.
                        # Also need to update the heap so that other threads don't select
                        # the GPU we just filled.
                        GPUStates.STATES[gid]['memoryUsed'] += requested_memory
                        GPUStates.STATES[gid]['memoryFree'] -= requested_memory
                        GPUStates.GPU_QUEUE.insert(gid, GPUStates.STATES[gid]['memoryUsed'])

                # Check for other instances of this unprofiled job type.
                # We shouldn't launch this job until we get a profile.
                # NOTE: Single dict op should be inherently thread-safe.
                elif JobStates.ACTIVE_JOBS.get(jobstr):
                    errorcode = ServiceErrorCode.WAITING_FOR_JOB_PROFILE
                
                # Artificially mark this device as fully allocated to avoid running
                # other new profile jobs on it.
                elif is_new_job_type:
                    GPUStates.STATES[gid]['memoryFree'] = 0

                    # Add a device lock that will persist until a profile is sent back.
                    # This is to prevent this device from being used for any other jobs
                    # since we don't know how many resources are required yet.
                    # NOTE: This device lock can only removed in the CompleteJob service.
                    GPUStates.PROFILING_GROUP[gid] = jobstr

    # If anything went wrong, set the GPU ID to something unreachable.
    if errorcode != ServiceErrorCode.OK:
        gid = -1

    return protos.GPU(
        gpu_id=gid, 
        errorcode=errorcode.value)


# TODO: These need to be methods in grn.core.globals
def update_job_state(jobstr: str, state) -> None:
    # Accumulate the max resource consumption.
    with JobStates.LOCK:
        JobStates.JOB_TYPES[jobstr] = max(
            JobStates.JOB_TYPES.get(jobstr, state),
            state)


def push_active_job(jobstr: str) -> None:
    with JobStates.LOCK:
        JobStates.ACTIVE_JOBS[jobstr] = JobStates.ACTIVE_JOBS.get(jobstr, 0) + 1


def pop_active_job(jobstr: str) -> None:
    with JobStates.LOCK:
        JobStates.ACTIVE_JOBS[jobstr] = JobStates.ACTIVE_JOBS.get(jobstr, 1) - 1


class GPUMasterServicer(services.GPUMasterServicer):
    """
    """
    def __init__(self, max_num_jobs: Optional[int] = None):
        # NOTE: Don't really need to clear these, but doing so provides an explicit
        #       in-code reminder that GPUMasterServicer uses these globals.
        JobStates.JOB_TYPES.clear()
        JobStates.ACTIVE_JOBS.clear()
        if (max_num_jobs is not None) and (max_num_jobs <= 0):
            raise ValueError('max_num_jobs must be a positive integer')
        JobStates.MAX_NUM_JOBS = max_num_jobs

    def RequestGPU(self, request: protos.JobType, context) -> protos.GPU:
        """Service that client invokes to obtain GPU resources.

        If this is a new job type, the GPU ID provisioned is guaranteed 
        to not have another profiling job running on it.

        If this is a profiled job type, then a GPU ID will be doled out
        according to the cached resource profile. If the resource request
        exceeds available resources, then wait until the request can be
        fulfilled.
        """
        log.debug(f'[RequestGPU] request\n{request}')
        jobstr = request.jobstr
        resource_policy = ResourcePolicy(request.resource_policy)
        gpu = get_next_available_gpu(jobstr, resource_policy)
        
        errorcode = ServiceErrorCode(gpu.errorcode)
        if errorcode == ServiceErrorCode.OK:
            push_active_job(jobstr)
            log.debug(f'[RequestGPU] serving GPU ID {gpu.gpu_id}')

        return gpu

    def CompleteJob(self, request: protos.JobProfile, context) -> protos.Empty:
        """Service that a client uses to send a completed job profile.
        If this is a new job type, then profile results will be cached
        and used to allocate resources to future jobs of this type.
        """
        log.debug(f'[CompleteJob] request\n{request}')
        jobstr = request.jobtype.jobstr
        # NOTE: Single python dict ops should be inherently thread-safe, so this
        #       should be okay.
        is_new_job_type = (JobStates.JOB_TYPES.get(jobstr) is None)

        # # FIXME: There is a problem where the SingleGPUMonitor will report the full
        # #        GPU usage across all jobs, which can lead to incorrect estimates of
        # #        resource consumption for job types.
        # #        As such, do not update job state continuously.
        # #        This means we have to rely on accurate profiling of the first job.
        # if request.succeeded:
        #     update_job_state(jobstr, state=request.max_gpu_memory_used)
        pop_active_job(jobstr)

        # When a job receives a profile, we should signal threads that are waiting on 
        # job profiles to attempt to resolve their resource requests.
        if is_new_job_type and request.succeeded:
            # Remove the device lock since profiling has finished.
            assert request.gpu.gpu_id in GPUStates.PROFILING_GROUP
            assert GPUStates.PROFILING_GROUP[request.gpu.gpu_id] == jobstr
            del GPUStates.PROFILING_GROUP[request.gpu.gpu_id]

            update_job_state(jobstr, state=request.max_gpu_memory_used)

        return protos.Empty()

    def JobTypeExists(self, request: protos.JobType, context) -> protos.BoolResponse:
        """Service that simply checks if a job type has already been profiled.
        """
        return protos.BoolResponse(
            response=(JobStates.JOB_TYPES.get(request.jobstr) is not None))


def __server_shutdown_sig_handler(*args) -> None:
    log.info('[GPUMaster] Cleaning up...')
    if os.path.exists(GRN_SERVER_TMP_INFO_PATH):
        os.remove(GRN_SERVER_TMP_INFO_PATH)
    sys.exit()


def serve(debug=False, max_workers=10):
    # TODO: Implement multiple servers at once without needing separate system environments (e.g. virtual environment, docker container).
    if os.path.isfile(GRN_SERVER_TMP_INFO_PATH):
        raise SystemError(
            f'GPU Master is already running! Shut down current server process before launching a new one.')

    signal.signal(signal.SIGINT, __server_shutdown_sig_handler)
    signal.signal(signal.SIGTERM, __server_shutdown_sig_handler)

    # Obtain a random free port from the OS and cache it to a secret file.
    # This file will be removed once the server shuts down.
    port = find_free_port()
    with open(GRN_SERVER_TMP_INFO_PATH, 'w') as grn_file:
        grn_file.write(f'{port}')

    with GPUMonitor(delay=0.1):
        server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers))
        services.add_GPUMasterServicer_to_server(GPUMasterServicer(), server)
        server.add_insecure_port(f'[::]:{port}')
        server.start()
        server.wait_for_termination()

    # Just in case.
    __server_shutdown_sig_handler()