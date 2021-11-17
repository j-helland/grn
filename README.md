# GPU Resource Negotiator (GRN)

GRN is a tool for evenly distributing N-many jobs across M-many GPUs on a single machine.
Its core assumption is that resource consumption for a job can only be determined at runtime.

## Main features
- **Minimal API** with almost no changes required to existing code. 
  Decorate functions with `@grn.job`.
  That's it!
- **Naturally composable** with other workflow and job orchestration tools.


# Examples
Working examples are maintained in the `examples/` directory if you want to jump straight in.
Consider starting with `examples/mnist`, which will give you a sense for how GRN can naturally compose with other workflow and job orchestration tools.


# Installation
- Run `pip install gpu-resource-negotiator`


# Usage
Decorate functions that require load balancing
```python
import grn

@grn.job()
def job():
    import torch
    x = torch.randn(1000, device='cuda:0')

if __name__ == '__main__':
    job()
```

If you have a long-running job that is not suitable for quick profiling, you may optionally define a lighter-weight version that will be profiled instead.
It is on you to ensure that this lightweight version is an accurate reflection of the
main job.
```python
import grn

@grn.job()
def long_job(sleep: int):
    import torch
    import time
    x = torch.randn(1000, device='cuda:0')
    time.sleep(sleep)

@long_job.profile
def short_job(*args):
    import torch
    import time
    x = torch.randn(1000, device='cuda:0')

if __name__ == '__main__':
    # short_job will run first to gather the resource profile.
    # Then long_job will run automatically.
    # The arguments passed to long_job here will also be passed to short_job.
    long_job(10)
```

## **Combined server/client invocation**
In your terminal, run.
```shell
$ grn-run "python job.py"
```
This command simply runs `grn-start` and then runs the command `python job.py` in sequence.
You can pass multiple commands if you want.
```shell
$ grn-run "python job.py" "python job.py"
```

## **Separate server/client invocation**
You can instead run the server by invoking in a separate terminal
```shell
$ grn-start

  ____  ____   _   _        ____  ____   _   _           __  __              _              
 / ___||  _ \ | \ | |  _   / ___||  _ \ | | | |         |  \/  |  __ _  ___ | |_  ___  _ __ 
| |  _ | |_) ||  \| | (_) | |  _ | |_) || | | |  _____  | |\/| | / _` |/ __|| __|/ _ \| '__|
| |_| ||  _ < | |\  |  _  | |_| ||  __/ | |_| | |_____| | |  | || (_| |\__ \| |_|  __/| |   
 \____||_| \_\|_| \_| (_)  \____||_|     \___/          |_|  |_| \__,_||___/ \__|\___||_|   


[11/12/21 20:16:35] INFO     Managing GPUs {0, 1, 2, 3}         gpu_monitors.py:38
```
and then in the original terminal
```shell
$ python job.py
```
Note that no `CUDA_VISIBLE_DEVICES` flag was used here.
If you want to restrict the possible GPUs that your job can run on, simply launch the server with a restricted range
```shell
$ CUDA_VISIBLE_DEVICES=0,1 grn-start

  ____  ____   _   _        ____  ____   _   _           __  __              _              
 / ___||  _ \ | \ | |  _   / ___||  _ \ | | | |         |  \/  |  __ _  ___ | |_  ___  _ __ 
| |  _ | |_) ||  \| | (_) | |  _ | |_) || | | |  _____  | |\/| | / _` |/ __|| __|/ _ \| '__|
| |_| ||  _ < | |\  |  _  | |_| ||  __/ | |_| | |_____| | |  | || (_| |\__ \| |_|  __/| |   
 \____||_| \_\|_| \_| (_)  \____||_|     \___/          |_|  |_| \__,_||___/ \__|\___||_|   
                                                                                    

[11/12/21 20:16:35] INFO     Managing GPUs {0, 1}               gpu_monitors.py:38
```
The load balancer will do the rest of the work to make sure that `job.py` will only run on a GPU in the set `{0, 1}`.


## Use with Docker

- **TL;DR**: Use the `--pid=host` argument when running the container.

To use GRN inside of a docker container, you need to run the container with the `--pid=host` flag, which disables namespace isolation of the container from the host.

THIS IS NOT SECURE. However, this security hole exists regardless when you use NVIDIA drivers.

NVIDIA drivers do not support Docker's namespace isolation, meaning that e.g. inside your container, calling `nvidia-smi` will give different PID values than the ones you see calling `top`.
This is because NVIDIA drivers directly access the host namespace regardless of your container.
For GRN to maintain accurate profiles of your jobs as they run, it relies on NVML calls which bypass container namespace isolation by way of the NVIDIA driver.
Without the same access to the host namespace, GRN inside the container cannot map the PID of your container's processes to the PIDs that NVML captures.


# More system details
The most important thing to know is that GRN assumes that the GPU resources required by a job are not known a priori -- whether this be due to dynamic computation graphs that cannot be analyzed before runtime or due to other complications.

This assumption leads to a couple of practical constraints and footguns.

- When you attempt to run a job that GRN has not encountered before, GRN will profile it in a blocking manner, meaning that any jobs of the same type that you attempt to run in the interim will wait until completion of the first.

  However, this ***does not*** mean that an unknown job of a different type will also be blocked.
  Rather, GRN can build profiles of multiple job types simultaneously.

## Potential Footguns
- GRN only caches profiles for **jobs that succeed**.
  If you have a large number of jobs that all fail at some midway point, they will all run one-at-a-time because GRN will never recognize any of them as known job types.
- You can explicitly tell GRN to use one job to build a profile for another. 
  If you use this feature, it is **up to you** to ensure that this other job accurately reflects the resource usage of the first.
  
  - If the secondary job overestimates resource consumption, you will not be able to maximize throughput.
  - If the secondary job underestimates resource consumption, jobs may fail unexpectedly due to CUDA out of memory errors.

## Job types
Currently, job types are a loosely defined concept for GRN.
It is assumed that 
It is assumed that jobs that share the same function signature are of the same type.
This means that a job type is sufficiently specified by a descriptor string of the form
```python
"tests/gpu_job_test.py::gpu_job"
```
This descriptor string is automatically built when you use the `@grn.job` decorator.
