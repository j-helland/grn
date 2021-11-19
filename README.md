# GPU Resource Negotiator (GRN)

GRN is a tool for dispatching N-many jobs onto M-many GPUs on a single machine, with support for multiple jobs on a single GPU.
Its core assumption is that resource consumption for a job can only be determined at runtime.

## Main features
- **Minimal API** with almost no changes required to existing code. 
  Decorate functions with `@grn.job`.
  That's it!
- **Naturally composable** with other workflow and job orchestration tools.
- **Framework agnostic:** PyTorch, Tensorflow, JAX -- use whatever!

## Limitations
- Only NVIDIA GPUs are currently supported.
- Multi-GPU jobs (e.g. PyTorch's [`DistributedDataParallel`](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)) are not currently supported.


# Examples
Working examples are maintained in the `examples/` directory if you want to jump straight in.
Consider starting with `examples/mnist`, which will give you a sense for how GRN can naturally compose with other workflow and job orchestration tools.

Table of Contents:
- [Pydoit example](examples/mnist/): coordinate the simultaneous training of MNIST models in PyTorch, Tensorflow, and JAX.


# Installation
- Run `pip install gpu-resource-negotiator`


# Usage
## **Combined server/client invocation**
The simplest way to use GRN is via the `grn-run` command, which is automatically added to your path on installation.
This approach requires no code changes.

Suppose you have a simple GPU script
```python
import argparse

def job(wait):
    import time
    import torch
    x = torch.randn(1024, device='cuda:0')

    if wait:
        time.sleep(wait)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--wait', type=float, default=None)
    args = parser.parse_args()

    job(args.wait)
```

In your terminal,
```shell
$ grn-run \
  "python job.py" \
  "python job.py" \
  "python job.py" \
  "python job.py --wait 5.0"
```
Under the hood, this is what happens:
1. `grn-start` is invoked to start the GPU Master.
2. Each `python job.py` instance requests GPU resources from GRN. 
    - One instance of each job type is allowed to run.
      There are two distinct jobs: `python job.py` and `python job.py --wait 5.0`, so two jobs will run simultaneously and send resource profiles to the GPU Master once they complete. 
    - The remaining two instances of `python job.py` are blocked until a resource profile for their job type is sent to the GPU Master.
      Once the profile is available, the GPU Master will distribute both jobs across GPUs according to the default resource policy `'spread'`, which evenly balances the GPUs.
3. Once all jobs complete, the GPU Master shuts down.

By default, GRN will manage all available GPUs on the machine.
You can restrict this using `CUDA_VISIBLE_DEVICES` e.g. `CUDA_VISIBLE_DEVICES=0,1 grn-run ...` to restrict GRN to only use the first two GPUs.

## **Separate server/client invocation**
The other way to use GRN is by launching the GPU Master as a standalone process.
This approach requires some small code changes, but gives you full control over the following:
- Which jobs have a particular job type. 
- The resource policy for each job type. You can spread out big jobs across GPUs while concentrating small jobs together.
- How your jobs are profiled. You can define quick auxiliary functions that will be used to profile resource requirements.
- The lifetime of the GPU Master. This means that job type profiles can be maintained beyond a single batch of jobs.

Currently, the way to hook up GPU jobs to GRN is with the `@grn.job` decorator.

**WARNING:** Do not use `grn-run` on scripts that use the `@grn.job` decorator.

`./job.py`:
```python
import grn

@grn.job()
def job():
    import torch
    x = torch.randn(1024, device='cuda:0')

if __name__ == '__main__':
    job()
```

### **Job types**
The job type in the example above will be automatically derived as `job.py::job` -- a combination of the function name and the file in which it is declared.
You can instead specify your own job type via `@grn.job(jobstr='my job type')`.

### **Resource policies**
You can control which resource allocation policy is used at a job-by-job level.
Currently, the two policies available are
1. `'spread'`: Allocate this job type to the GPU with the most resources available at request time.
2. `'pack'`: Allocate this job type to the GPU with the least amount of resources that can still satisfy its resource requirements at request time.
  NOTE: This does not necessarily mean that all jobs with the `'pack'` policy will cluster together on the same device(s).

You can specify the policy via e.g. `@grn.job(resource_policy='pack')`.

The default policy is `'spread'`.

### **Custom profiling**
If you have a long-running job that is not suitable for quick profiling, you may optionally define a lighter-weight version that will be profiled instead.
It is an exercise for the reader to ensure that this lightweight version is an accurate reflection of the main job.
```python
import grn

@grn.job()
def long_job(sleep: int):
    import torch
    import time
    x = torch.randn(1024, device='cuda:0')
    time.sleep(sleep)

@long_job.profile
def short_job(*args):
    import torch
    import time
    x = torch.randn(1024, device='cuda:0')

if __name__ == '__main__':
    # short_job will run first to gather the resource profile.
    # Then long_job will run automatically.
    # The arguments passed to long_job here will also be passed to short_job.
    long_job(10)
```

### **Launching the GPU Master**
Launch the server in a separate terminal in the same localhost network:
```shell
$ grn-start

   ____ _     ____         ____ ____  _   _           __  __           _
  / ___| |   | __ )   _   / ___|  _ \| | | |         |  \/  | __ _ ___| |_ ___ _ __
 | |  _| |   |  _ \  (_) | |  _| |_) | | | |  _____  | |\/| |/ _` / __| __/ _ \ '__|
 | |_| | |___| |_) |  _  | |_| |  __/| |_| | |_____| | |  | | (_| \__ \ ||  __/ |
  \____|_____|____/  (_)  \____|_|    \___/          |_|  |_|\__,_|___/\__\___|_|

[11/12/21 20:16:35] INFO     Managing GPUs {0, 1, 2, 3}         gpu_monitors.py:38
```
and then in your original terminal
```shell
$ python job.py
```
which will automatically send a request to the server for resources.

### **Restricting devices**
Note that no `CUDA_VISIBLE_DEVICES` flag was used in the example above.
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
GRN will do the rest of the work to make sure that `job.py` will only run on a GPU in the set `{0, 1}`.


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
