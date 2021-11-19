import os
import time
import psutil
from threading import Thread
import logging

import nvidia_smi as nvml

from grn.core.globals import GPUStates


log = logging.getLogger(__file__)


__all__ = ['GPUMonitor', 'SingleGPUProcessMonitor']


class GPUMonitor(Thread):
    """
    """
    def __init__(self, delay=1):
        super().__init__()

        nvml.nvmlInit()
        try:
            self.num_gpus_available = nvml.nvmlDeviceGetCount()
        except nvml.NVMLError:
            raise LookupError(
                f'Could not acquire device count, meaning there are no NVIDIA devices available.')

        # The GPU master must respect external CUDA_VISIBLE_DEVICES restrictions.
        self.cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES')
        if self.cuda_visible_devices is not None:
            self.cuda_visible_devices = set(int(d) for d in self.cuda_visible_devices.split(','))
        else:
            self.cuda_visible_devices = set(range(self.num_gpus_available))

        log.info(f'Managing GPUs {self.cuda_visible_devices}\n')
 
        try:
            self.gpu_handles = {
                idx: nvml.nvmlDeviceGetHandleByIndex(idx) 
                for idx in range(self.num_gpus_available)
                if idx in self.cuda_visible_devices}
        except nvml.NVMLError:
            raise LookupError(
                f'Could not acquire one or more GPU device handles.')

        self.stopped = False
        self.delay = delay
        self.start()

    def _update_gpu_states(self, states):
        # The lock is probably necessary here because of the sequence of appends.
        # We don't want to get interrupted by the server trying to heappop a gpu.
        with GPUStates.LOCK:
            gpu_list = []
            for gid in states:
                GPUStates.STATES[gid] = states[gid]
                gpu_list.append( (gid, states[gid][GPUStates.TARGET_RESOURCE]) )

            # Heapify at the end to for O(n) instead of O(n log n).
            GPUStates.GPU_QUEUE.clear()
            GPUStates.GPU_QUEUE.extend(gpu_list)

        GPUStates.STATES_INIT.set()

    def run(self):
        while not self.stopped:
            try:
                states = {
                    idx: (
                        nvml.nvmlDeviceGetMemoryInfo(h), 
                        nvml.nvmlDeviceGetUtilizationRates(h)) 
                    for idx, h in self.gpu_handles.items()}
            except nvml.NVMLError:
                break

            states = {
                idx: dict(
                    memoryUsed=info[0].used,
                    memoryTotal=info[0].total,
                    memoryFree=info[0].free,
                    load=info[1].gpu) 
                for idx, info in states.items()}
            self._update_gpu_states(states)

            time.sleep(self.delay)

    def __enter__(self):
        return self
    
    def __exit__(self, type, value, traceback):
        nvml.nvmlShutdown()
        self.stopped = True


class SingleGPUProcessMonitor(Thread):
    """Separate thread for monitoring usage stats of a target GPU.
    """
    def __init__(self, gpu_id: int, pid: int, delay: float):
        super().__init__()
        nvml.nvmlInit()
        try:
            self.gpu_handle = nvml.nvmlDeviceGetHandleByIndex(gpu_id)
        except nvml.NVMLError:
            raise LookupError(
                f'Could not acquire device handle for GPU with index {gpu_id}.')

        self.stopped = False
        self.delay = delay
        self.gpu_id = gpu_id
        self.proc = psutil.Process(pid)
        self.max_mem_used = 0
        self.daemon = True
        self.start()

    def run(self):
        while not self.stopped:
            used_memory = 0
            target_procs = {p.pid: p for p in self.proc.children(recursive=True)}
            target_procs[self.proc.pid] = self.proc
            try:
                for proc in nvml.nvmlDeviceGetComputeRunningProcesses(self.gpu_handle):
                    if proc.pid in target_procs:
                        used_memory += proc.usedGpuMemory
            except nvml.NVMLError:
                break

            if used_memory > 0:
                # Aggregate peak usage
                self.max_mem_used = max(self.max_mem_used, used_memory)

            time.sleep(self.delay)
    
    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        nvml.nvmlShutdown()
        self.stopped = True
