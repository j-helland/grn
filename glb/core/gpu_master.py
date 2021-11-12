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
import time
import heapq
from concurrent import futures
import threading
from threading import Thread
import signal
import logging

import grpc

import glb.utils
import glb.grpc_resources.master_pb2 as protos
import glb.grpc_resources.master_pb2_grpc as services
from glb.core.constants import GLB_SERVER_TMP_INFO_PATH, ServiceErrorCodes
from glb.core.globals import GPUStates, JobStates

from typing import Optional


log = logging.getLogger(__file__)


__all__ = ['serve']


class BackgroundEventTrigger(Thread):
    """
    """
    def __init__(self, event: threading.Event, delay=5):
        super().__init__()
        self.event = event
        self.stopped = False
        self.delay = delay
        self.start()

    def run(self):
        while not self.stopped:
            time.sleep(self.delay)
            self.event.set()

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.stopped = True


def get_next_available_gpu(jobstr: str) -> protos.GPU:
    GPUStates.STATES_INIT.wait()

    with GPUStates.LOCK:
        # FIXME: We assume GPU homogeneity -- all GPUs have identical architecture.
        # TODO: Implement a segmented heap to group homogenous hardware together.
        #       Then we can have a resource heap for each bin.

        # Cycle through GPUs in order of most memory available.
        # Keep looking until we find one that isn't device locked.
        # group = heapq.nsmallest(len(GPUStates.HEAP), GPUStates.HEAP)
        i = 0
        group = heapq.nsmallest(1, GPUStates.HEAP)
        gid = group[0][1]
        while (i < len(GPUStates.HEAP)) and (gid in GPUStates.PROFILING_GROUP):
            # Keep doubling the number of elements to look at so we can avoid
            # the O(log k) cost most of the time.
            group = heapq.nsmallest(
                min(len(group) * 2, len(GPUStates.HEAP)), 
                GPUStates.HEAP)
            i += 1
            gid = group[i][1]

        # If we reached the end of the heap, there must be no devices that aren't locked.
        if i == len(GPUStates.HEAP):
            errorcode = ServiceErrorCodes.WAITING_FOR_JOB_PROFILE
        
        else:
            mem_free = GPUStates.STATES[gid]['memoryFree']
            mem_total = GPUStates.STATES[gid]['memoryTotal']
            # NOTE: Single python dict op should be inherently thread-safe.
            requested_memory = JobStates.JOB_TYPES.get(jobstr)
            log.debug(f'{mem_free}, {mem_total}, {requested_memory}')

            errorcode = ServiceErrorCodes.OK
            if requested_memory is not None:
                # This job can never be completed.
                # FIXME: This state should never be reached.
                if requested_memory > mem_total:
                    errorcode = ServiceErrorCodes.EXCEEDS_TOTAL_MEMORY
                    raise MemoryError(
                        f'[RequestGPU] service encountered errorcode {errorcode}.'
                        'Something has gone terribly wrong...')

                # This job could be completed, but not right now.
                elif requested_memory > mem_free:
                    errorcode = ServiceErrorCodes.EXCEEDS_CURRENT_MEMORY

                # We need to update GPU state right now so that other threads
                # will immediately become aware of new resource constraints.
                # Also need to update the heap so that other threads don't select
                # the GPU we just filled.
                else:
                    GPUStates.STATES[gid]['memoryFree'] -= requested_memory
                    heapq.heapreplace(
                        GPUStates.HEAP, 
                        (mem_total - GPUStates.STATES[gid]['memoryFree'], gid))

            # Check for other instances of this unprofiled job type.
            # We shouldn't launch this job until we get a profile.
            # NOTE: Single dict op should be inherently thread-safe.
            elif JobStates.ACTIVE_JOBS.get(jobstr):
                errorcode = ServiceErrorCodes.WAITING_FOR_JOB_PROFILE
            
            # Artificially mark this device as fully allocated to avoid running
            # other new profile jobs on it.
            elif JobStates.JOB_TYPES.get(jobstr) is None:
                GPUStates.STATES[gid]['memoryFree'] = 0
                heapq.heapreplace(
                    GPUStates.HEAP, 
                    (mem_total, gid))

                # Add a device lock that will persist until a profile is sent back.
                # This is to prevent this device from being used for any other jobs
                # since we don't know how many resources are required yet.
                # NOTE: This device lock can only removed in the CompleteJob service.
                GPUStates.PROFILING_GROUP[gid] = jobstr

        # If anything went wrong, set the GPU ID to something unreachable.
        if errorcode != ServiceErrorCodes.OK:
            gid = -1

    return protos.GPU(
        gpu_id=gid, 
        errorcode=errorcode.value)


# TODO: These need to be methods in glb.core.globals
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
        JobStates.READY.set()

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
        gpu = get_next_available_gpu(jobstr)
        
        # If this request is within total memory limits but memory is not currently
        # available, then we need to wait until a job completes to check again.
        # TODO: This could result in endlessly spinning jobs.
        #       A job queue probably makes sense so we make sure that jobs
        #       don't get left hanging. This shold only be a problem when other
        #       users / instances of this tool are using this machine.
        errorcode = ServiceErrorCodes(gpu.errorcode)
        while ( (errorcode == ServiceErrorCodes.EXCEEDS_CURRENT_MEMORY) or  # FIXME: this error code should never occur
                (errorcode == ServiceErrorCodes.WAITING_FOR_JOB_PROFILE) ):
            log.debug(f'{errorcode}: waiting...')
            JobStates.READY.clear()
            JobStates.READY.wait()
            gpu = get_next_available_gpu(jobstr)
            errorcode = ServiceErrorCodes(gpu.errorcode)

        if errorcode == ServiceErrorCodes.OK:
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
        new_job_type = (JobStates.JOB_TYPES.get(jobstr) is None)

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
        if new_job_type and request.succeeded:
            # Remove the device lock since profiling has finished.
            assert request.gpu.gpu_id in GPUStates.PROFILING_GROUP
            assert GPUStates.PROFILING_GROUP[request.gpu.gpu_id] == jobstr
            del GPUStates.PROFILING_GROUP[request.gpu.gpu_id]

            update_job_state(jobstr, state=request.max_gpu_memory_used)
            JobStates.READY.set()

        return protos.Empty()

    def JobTypeExists(self, request: protos.JobType, context) -> protos.BoolResponse:
        """Service that simply checks if a job type has already been profiled.
        """
        return protos.BoolResponse(
            response=(JobStates.JOB_TYPES.get(request.jobstr) is not None))


def __server_shutdown_sig_handler(*args) -> None:
    log.info('[GPUMaster] Cleaning up...')
    if os.path.exists(GLB_SERVER_TMP_INFO_PATH):
        os.remove(GLB_SERVER_TMP_INFO_PATH)
    sys.exit()


def serve(debug=False):
    signal.signal(signal.SIGINT, __server_shutdown_sig_handler)
    signal.signal(signal.SIGTERM, __server_shutdown_sig_handler)

    # Obtain a random free port from the OS and cache it to a secret file.
    # This file will be removed once the server shuts down.
    port = glb.utils.find_free_port()
    with open(GLB_SERVER_TMP_INFO_PATH, 'w') as glb_file:
        glb_file.write(f'{port}')

    with BackgroundEventTrigger(JobStates.READY, delay=2), glb.utils.GPUMonitor(delay=0.1):
        server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
        services.add_GPUMasterServicer_to_server(GPUMasterServicer(), server)
        server.add_insecure_port(f'[::]:{port}')
        server.start()
        server.wait_for_termination()

    # Just in case.
    __server_shutdown_sig_handler()