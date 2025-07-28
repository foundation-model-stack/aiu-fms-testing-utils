# aiu-fms-testing-utils

## Setup your environment

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
export FLEX_DEVICE=VFIO
```

Optional envars to supress debugging output:
```shell
export DTLOG_LEVEL=error
export TORCH_SENDNN_LOG=CRITICAL
export DT_DEEPRT_VERBOSE=-1
```

## Example of using Foundation Model Stack (FMS) on AIU hardware

Let's look at a simple example of running [FMS](https://github.com/foundation-model-stack/foundation-model-stack) on AIU. While you can use a variety of models, in this example, we will use the Llama-194m decoder model.

The first step is to make sure that the required libraries are available. For our example, we will use the following libraries.
```
import math
import os
import torch

from aiu_fms_testing_utils.utils import warmup_model
from aiu_fms_testing_utils.utils.aiu_setup import dprint
from fms.models import get_model
from fms.utils.generation import generate, pad_input_ids
from torch_sendnn import torch_sendnn
from transformers import AutoTokenizer
```

For our example decoder model, we will set compilation mode to offline_decoder instead of using the default.
```
os.environ.setdefault("COMPILATION_MODE", "offline_decoder")
```

Now, add the model setup and tokenizer details.
```
model_path = "/home/devel/models/llama-194m" # Change the model path as needed
model = get_model(
    architecture="hf_pretrained",
    model_path=model_path,
    device_type="cpu",
    fused_weights=False,
)
model.eval()
torch.set_grad_enabled(False)
model.compile(backend="sendnn") # Compile with the AIU sendnn backend

# Tokenize
tokenizer = AutoTokenizer.from_pretrained(model_path)
```

Let's define our desired prompt.
```
template = "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{}\n\n### Response:"

prompt = template.format("Provide a list of instructions for preparing chicken soup.")

input_ids = tokenizer.encode(prompt, return_tensors="pt")
input_ids, extra_generation_kwargs = pad_input_ids([input_ids.squeeze(0)], min_pad_length=math.ceil(input_ids.size(1)/64) * 64)

# only_last_token optimization
extra_generation_kwargs["only_last_token"] = True

# Set a desired number
max_new_tokens = 16
```

We are ready to generate response.
```
warmup_model(model, input_ids, max_new_tokens=max_new_tokens, **extra_generation_kwargs)

# Generate model response
result = generate(
    model,
    input_ids,
    max_new_tokens=max_new_tokens,
    use_cache=True,
    max_seq_len=model.config.max_expected_seq_len,
    contiguous_cache=True,
    do_sample=False,
    extra_kwargs=extra_generation_kwargs,
)
```

Optionally, we will also print the output.
```
def print_result(result):
    output_str = tokenizer.convert_tokens_to_string(
        tokenizer.convert_ids_to_tokens(result)
    )
    dprint(output_str)
    print("...")

for i in range(result.shape[0]):
    print_result(result[i])
  ```

The output should be similar to the one below.
```
### Instruction:
Provide a list of instructions for preparing chicken soup.

### Response: Chicken soup is a popular soup that is made from chicken, vegetables, and other ...
```

You can try this example by simply copying and pasting the code and running it as a Python script.

If you want to run different models, the `inference.py` script is what you want to use. You can find it, as well as other example scripts, in the scripts folder. You will get a similar output to the example code provided above by running `inference.py` with Llama-194m as below:
```bash
$ python3 inference.py --architecture=hf_pretrained --model_path=/home/devel/models/llama-194m --tokenizer=/home/devel/models/llama-194m --unfuse_weights --min_pad_length 64 --device_type=aiu --compile --compile_dynamic
```

More examples are provided below to run various Llama and Granite models. A slimmed down version of the Big Toy model is also provided to demonstrate how to run a tensor parallel model with the FMS on AIU hardware.

## More Examples

 Tensor parallel execution is only supported on the AIU through the [Foundation Model Stack](https://github.com/foundation-model-stack/foundation-model-stack).

The `--nproc-per-node` command line option controls the number of AIUs to use (number of parallel processes).

### Small Toy

The `small-toy.py` is a slimmed down version of the Big Toy model. The purpose of this model is to demostrate how to run a tensor parallel model with the FMS on AIU hardware.

```bash
cd ${HOME}/aiu-fms-testing-utils/scripts

# 1 AIU (sequential)
# Inductor (CPU) backend (default)
torchrun --nproc-per-node 1 ./small-toy.py
# AIU backend
torchrun --nproc-per-node 1 ./small-toy.py --backend aiu

# 2 AIUs (tensor parallel)
# Inductor (CPU) backend (default)
torchrun --nproc-per-node 2 ./small-toy.py
# AIU backend
torchrun --nproc-per-node 2 ./small-toy.py --backend aiu
```

Example Output

```console
shell$ torchrun --nproc-per-node 4 ./small-toy.py --backend aiu
------------------------------------------------------------
0 / 4 : Python Version  : 3.11.7
0 / 4 : PyTorch Version : 2.2.2+cpu
0 / 4 : Dynamo Backend  : aiu -> sendnn
0 / 4 : PCI Addr. for Rank 0 : 0000:bd:00.0
0 / 4 : PCI Addr. for Rank 1 : 0000:b6:00.0
0 / 4 : PCI Addr. for Rank 2 : 0000:b9:00.0
0 / 4 : PCI Addr. for Rank 3 : 0000:b5:00.0
------------------------------------------------------------
0 / 4 : Creating the model...
0 / 4 : Compiling the model...
0 / 4 : Running model: First Time...
0 / 4 : Running model: Second Time...
0 / 4 : Done
```

### LLaMA/Granite
```bash
export DT_OPT=varsub=1,lxopt=1,opfusion=1,arithfold=1,dataopt=1,patchinit=1,patchprog=1,autopilot=1,weipreload=0,kvcacheopt=1,progshareopt=1

# run 194m on AIU
python3 inference.py --architecture=hf_pretrained --model_path=/home/senuser/llama3.194m --tokenizer=/home/senuser/llama3.194m --unfuse_weights --min_pad_length 64 --device_type=aiu --max_new_tokens=5 --compile --default_dtype=fp16 --compile_dynamic

# run 194m on CPU
python3 inference.py --architecture=hf_pretrained --model_path=/home/senuser/llama3.194m --tokenizer=/home/senuser/llama3.194m --unfuse_weights --min_pad_length 64 --device_type=cpu --max_new_tokens=5 --default_dtype=fp32

# run 7b on AIU
python3 inference.py --architecture=hf_pretrained --model_path=/home/senuser/llama2.7b --tokenizer=/home/senuser/llama2.7b --unfuse_weights --min_pad_length 64 --device_type=aiu --max_new_tokens=5 --compile --default_dtype=fp16 --compile_dynamic

# run 7b on CPU
python3 inference.py --architecture=hf_pretrained --model_path=/home/senuser/llama2.7b--tokenizer=/home/senuser/llama2.7b --unfuse_weights --min_pad_length 64 --device_type=cpu --max_new_tokens=5 --default_dtype=fp32

# run gpt_bigcode (granite) 3b on AIU
python3 inference.py --architecture=gpt_bigcode --variant=ibm.3b --model_path=/home/senuser/gpt_bigcode.granite.3b/*00002.bin --model_source=hf --tokenizer=/home/senuser/gpt_bigcode.granite.3b --unfuse_weights --min_pad_length 64 --device_type=aiu --max_new_tokens=5 --prompt_type=code --compile --default_dtype=fp16 --compile_dynamic

# run gpt_bigcode (granite) 3b on CPU
python3 inference.py --architecture=gpt_bigcode --variant=ibm.3b --model_path=/home/senuser/gpt_bigcode.granite.3b/*00002.bin --model_source=hf --tokenizer=/home/senuser/gpt_bigcode.granite.3b --unfuse_weights --min_pad_length 64 --device_type=cpu --max_new_tokens=5 --prompt_type=code --default_dtype=fp32
```

To try mini-batch, use `--batch_input`

For the validation script, here are a few examples:

```bash
export DT_OPT=varsub=1,lxopt=1,opfusion=1,arithfold=1,dataopt=1,patchinit=1,patchprog=1,autopilot=1,weipreload=0,kvcacheopt=1,progshareopt=1

# Run a llama 194m model, grab the example inputs in the script, generate validation tokens on cpu, validate token equivalency: 
python3 scripts/validation.py --architecture=hf_pretrained --model_path=/home/devel/models/llama-194m --tokenizer=/home/devel/models/llama-194m --unfuse_weights --batch_size=1 --min_pad_length=64 --max_new_tokens=10 --compile_dynamic

# Run a llama 194m model, grab the example inputs in a folder, generate validation tokens on cpu, validate token equivalency:
python3 scripts/validation.py --architecture=hf_pretrained --model_path=/home/devel/models/llama-194m --tokenizer=/home/devel/models/llama-194m --unfuse_weights --batch_size=1 --min_pad_length=64 --max_new_tokens=10 --prompt_path=/home/devel/aiu-fms-testing-utils/prompts/test/*.txt --compile_dynamic

# Run a llama 194m model, grab the example inputs in a folder, grab validation text from a folder, validate token equivalency (will only validate up to max(max_new_tokens, tokens_in_validation_file)):
python3 scripts/validation.py --architecture=hf_pretrained --model_path=/home/devel/models/llama-194m --tokenizer=/home/devel/models/llama-194m --unfuse_weights --batch_size=1 --min_pad_length=64 --max_new_tokens=10 --prompt_path=/home/devel/aiu-fms-testing-utils/prompts/test/*.txt --validation_files_path=/home/devel/aiu-fms-testing-utils/prompts/validation/*.txt --compile_dynamic

# Validate a reduced size version of llama 8b
python3 scripts/validation.py --architecture=hf_configured --model_path=/home/devel/models/llama-8b --tokenizer=/home/devel/models/llama-8b --unfuse_weights --batch_size=1 --min_pad_length=64 --max_new_tokens=10 --extra_get_model_kwargs nlayers=3 --compile_dynamic
```

To run a logits-based validation, pass `--validation_level=1` to the validation script. This will check for the logits output to match at every step of the model through cross-entropy loss.
You can control the acceptable threshold with `--logits_loss_threshold`

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
