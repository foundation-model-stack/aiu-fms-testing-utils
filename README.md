# Installation Guide for AIU FMS Testing Utilities

This guide provides instructions for installing the `aiu-fms-testing-utils` package.

## Installation with CPU-Only PyTorch

To install the package with the CPU-only version of PyTorch, run the following command:

```shell
pip3 install aiu-fms-testing-utils --extra-index-url=https://download.pytorch.org/whl/cpu
```

## Installation with Default PyTorch

To install the package with the platform's default PyTorch version, execute:

```shell
pip3 install aiu-fms-testing-utils
```

## Verify the PyTorch Version

To ensure compatibility with aiu-fms-testing-utils, verify that the correct PyTorch version is installed.

1. Check the PyTorch Version

    ```shell
    pip show torch
    ```

    Expected Output for CPU-Only PyTorch:

    ```plain
    Name: torch
    Version: 2.7.1+cpu
    ```

    For the CPU-only version, the version string should include the +cpu postfix (e.g., 2.7.1+cpu).

2. Corrective Action for Default PyTorch

    If the installed PyTorch version does not include the +cpu postfix (e.g., 2.7.1 without +cpu), it indicates that the default PyTorch version (which may include CUDA support) was installed. To resolve this, uninstall torch and reinstall aiu-fms-testing-utils with the CPU-only version:

    ```shell
    pip3 uninstall torch -y
    pip3 install aiu-fms-testing-utils --extra-index-url=https://download.pytorch.org/whl/cpu
    ```

## Setting Up the Development Environment from Source

To set up the development environment for aiu-fms-testing-utils from source, follow these steps.

In this directory, checkout the Foundation Model Stack (FMS) and the FMS Model Optimizer:

```shell
git clone https://github.com/foundation-model-stack/foundation-model-stack.git
git clone https://github.com/foundation-model-stack/fms-model-optimizer.git
```

Install both FMS, FMS-Model-Optimizer and aiu-fms-testing-utils:

```shell
cd foundation-model-stack
pip install -e .
cd ..

cd fms-model-optimizer
pip install -e .
cd ..

pip install -e .
```

### Running in OpenShift

Use the `pod.yaml` file to get started with your OpenShift allocation

* Modify the `ibm.com/aiu_pf_tier0` values to indicate the number of AIUs that you want to use
* Modify the `namespace` to match your namespace/project (i.e., `oc project`)

Start the pod

```shell
oc apply -f pod.yaml
```

Copy this repository into the pod (includes scripts, FMS stack)

```shell
oc cp ${PWD} my-workspace:/tmp/
```

Exec into the pod

```shell
 oc rsh my-workspace bash -l
 ```

When you are finished, make sure to delete your pod:

```shell
oc delete -f pod.yaml
```

### Setup the environment in the container

Verify the AIU discovery has happened by looking for output like the following when you exec into the pod:

```console
---- IBM AIU Device Discovery...
---- IBM AIU Environment Setup... (Generate config and environment)
---- IBM AIU Devices Found: 2
------------------------
[1000760000@my-workspace ~]$  echo $AIU_WORLD_SIZE
2
```

Inside the container, setup envars to use the FMS:

```shell
export HOME=/tmp
cd ${HOME}/aiu-fms-testing-utils/foundation-model-stack/
# Install the FMS stack
pip install -e .
```

Run with AIU instead of, default, senulator.

```shell
export FLEX_COMPUTE=SENTIENT
export FLEX_DEVICE=PF
```

Optional envars to supress debugging output:

```shell
export DTLOG_LEVEL=error
export TORCH_SENDNN_LOG=CRITICAL
export DT_DEEPRT_VERBOSE=-1
```

## How to use Foundation Model Stack (FMS) on AIU hardware
The [scripts](https://github.com/foundation-model-stack/aiu-fms-testing-utils/tree/main/scripts) directory provides various scripts to use FMS on AIU hardware for many use cases. These scripts provide robust support for passing desired command line options for running encoder and decoder models along with other use cases. Refer to the documentation on [using different scripts](https://github.com/foundation-model-stack/aiu-fms-testing-utils/blob/main/scripts/README.md) for more details.

The [examples](https://github.com/foundation-model-stack/aiu-fms-testing-utils/tree/main/examples) directory provides small examples aimed at helping understand the general workflow of running a model using FMS on AIU hardware.

## Common Errors

### Pod connection error

Errors like the following often indicate that the pod has not started or is still in the process of starting.

```console
error: unable to upgrade connection: container not found ("my-pod")
```

Use `oc get pods` to check on the status. `ContainerCreating` indicates that the pod is being created. `Running` indicates that it is ready to use.

If there is an error the use `oc describe pod/my-workspace` to see a full diagnostic view. The `Events` list at the bottom will often let you know what the problem is.

### torchrun generic error

Below is the generic `torchrun` failed program trace. It is not helpful when trying to find the problem in the program. Instead look for the actual error message a little higher in the output trace.

```console
[2024-09-16 16:10:15,705] torch.distributed.elastic.multiprocessing.api: [ERROR] failed (exitcode: 1) local_rank: 0 (pid: 1479484) of binary: /usr/bin/python3
Traceback (most recent call last):
  File "/usr/local/bin/torchrun", line 8, in <module>
    sys.exit(main())
  File "/usr/local/lib64/python3.9/site-packages/torch/distributed/elastic/multiprocessing/errors/__init__.py", line 347, in wrapper
    return f(*args, **kwargs)
  File "/usr/local/lib64/python3.9/site-packages/torch/distributed/run.py", line 812, in main
    run(args)
  File "/usr/local/lib64/python3.9/site-packages/torch/distributed/run.py", line 803, in run
    elastic_launch(
  File "/usr/local/lib64/python3.9/site-packages/torch/distributed/launcher/api.py", line 135, in __call__
    return launch_agent(self._config, self._entrypoint, list(args))
  File "/usr/local/lib64/python3.9/site-packages/torch/distributed/launcher/api.py", line 268, in launch_agent
    raise ChildFailedError(
torch.distributed.elastic.multiprocessing.errors.ChildFailedError:
============================================================
./roberta.py FAILED
------------------------------------------------------------
Failures:
  <NO_OTHER_FAILURES>
------------------------------------------------------------
Root Cause (first observed failure):
[0]:
  time      : 2024-09-16_16:10:15
  host      : ibm-aiu-rdma-jjhursey
  rank      : 0 (local_rank: 0)
  exitcode  : 1 (pid: 1479484)
  error_file: <N/A>
  traceback : To enable traceback see: https://pytorch.org/docs/stable/elastic/errors.html
============================================================
```

### Additional warnings

You may see the following additional warnings/notices printed to the console. They are normal and expected at this point in time. The team will work on cleaning these up.

```console
CUDA extension not installed.
using tensor parallel
ignoring module=Embedding when distributing module
[WARNING] Keys from checkpoint (adapted to FMS) not copied into model: {'roberta.embeddings.token_type_embeddings.weight', 'lm_head.bias'}
```
