# MNIST Example

The purpose of this example is to show how to use GLB in composition with other job orchestration tools.
The tool of choice for this demo will be [doit](https://pydoit.org/), which is a Python DAG-based workflow specification and orchestration tool.

The details of doit are not important -- it's just another workflow tool.
All that matters is that doit will run jobs in a logical ordering.

## File organization and notable bits

- `tf_mnist_train.py` is a simple script that trains a small model on MNIST. 
    The details are not important, only the use of the `@gpu_load_balance` decorator:
    ```python
    @glb.gpu_load_balance()
    def train_job(epochs: int, batch_size: int) -> 'tf.keras.Model':
        # continued...
    ```
    Note that Tensorflow by default uses a GPU preallocation policy
    To get more accurate estimates of resource consumption for this job, we can set a dynamic growth policy.
    ```python
    physical_devices = tf.config.list_physical_devices('GPU') 
    for gpu_instance in physical_devices: 
        tf.config.experimental.set_memory_growth(gpu_instance, True)
    ```

- `torch_mnist_train.py` is an implementation of `tf_mnist_train.py` in PyTorch. Once again, the only thing that matters is the decorator usage:
    ```python
    @glb.gpu_load_balance()
    def train_job(epochs: int, batch_size: int) -> 'torch.nn.Module':   
        # continued...
    ```
    PyTorch by default uses a dynamic memory growth policy.

- `jax_mnist_train.py` is yet another implementation of `tf_mnist_train.py`, only this time using JAX as the numerical backend and FLAX as the deep learning framework.
    This is just to round out our example with a third job type.
    Similarly to Tensorflow, it behooves us tell JAX to use a dynamic growth policy rather than a preallocation policy.
    ```python
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = "false"
    ```

- `mnist_doit.py` is a minimal doit workflow specification containing task definitions of the form
    ```python
    def task_0():
        return {
            'doc': 'Tensorflow MNIST training 0',
            'file_dep': [ ],
            'targets': [ ],
            'actions': [['python', 'tf_mnist_train.py']],
            'clean': [clean_targets],
        }
    ```
    In this case, there are no dependencies between jobs, so they can all be run in parallel.

## Running the example

There are two ways you can run this example.
The first is the quick way, while the second requires an additional step but will provide you a better understanding of what the system is doing.

1. **Quickstart:** Run `glb-run "doit -d . -f example_doit.py -n 9"`.
    This starts the load balancer automatically and then invokes a doit command that launches all 9 training jobs in parallel.
    Your terminal window will fill with spam.

2. **In-depth:** Open a separate terminal session and run `glb-start`.
    You should see something like the following:
    ```
    $ glb-start

      ____ _     ____         ____ ____  _   _           __  __           _
     / ___| |   | __ )   _   / ___|  _ \| | | |         |  \/  | __ _ ___| |_ ___ _ __
    | |  _| |   |  _ \  (_) | |  _| |_) | | | |  _____  | |\/| |/ _` / __| __/ _ \ '__|
    | |_| | |___| |_) |  _  | |_| |  __/| |_| | |_____| | |  | | (_| \__ \ ||  __/ |
     \____|_____|____/  (_)  \____|_|    \___/          |_|  |_|\__,_|___/\__\___|_|

    [11/12/21 20:16:35] INFO     Managing GPUs {0, 1, 2, 3}         gpu_monitors.py:38
    ```
    In your original terminal, run the command `doit -d . -f example_doit.py -n 9` to get the following output.
    ```shell
    $ doit -d . -f example_doit.py -n 9
    .  0
    .  1
    .  2
    .  3
    .  4
    .  5
    .  6
    .  7
    .  8
    ```
    Now in your GLB terminal, `Ctrl-C` to shut down the server. 
    You should see
    ```shell
    [11/12/21 20:16:37] INFO     [GPUMaster] Cleaning up...         gpu_master.py:2
    $
    ```
    <!-- In your GLB terminal, you should quickly start to see output like
    ```shell
    [RequestGPU] request  jobstr: "tf_mnist_train.py::train_job"

    34089598976 34089730048 None
    [RequestGPU] request  jobstr: "tf_mnist_train.py::train_job"

    [RequestGPU] serving GPU ID 0
    34089598976 34089730048 None
    ServiceErrorCodes.WAITING_FOR_JOB_PROFILE: waiting...  
    ```
    Indicating that your jobs have launched. 
    In this snippet, we can see that two jobs of the same type were submitted and that the first one was given GPU 0 and the other one was blocked, waiting until the first job completes and sends back its profile.
    When a job does complete, you will see a message corresponding to the job profile
    ```shell
    [CompleteJob] request
    jobtype {
        jobstr: "tf_mnist_train.py::train_job"
    }
    gpu {
        gpu_id: 0
    }
    succeeded: true
    max_gpu_memory_used: 1157890048
    max_gpu_load: 20
    ``` -->
