# Small examples of using Foundation Model Stack (FMS) on AIU hardware

The [scripts](https://github.com/foundation-model-stack/aiu-fms-testing-utils/tree/main/scripts) directory provides robust scripts allowing users to pass various command-line options. You should use them according to your use case. However, considering they are robust and bigger, it can be difficult to follow the flow quickly. The examples provided here serve a short workflow, helping users quickly understand how to run FMS on AIU hardware.

We will walk through an example of running the IBM Granite model with the AIU backend.
The first step is to make sure that the required libraries are available, which includes [aiu-fms-testing-utils](https://github.com/foundation-model-stack/aiu-fms-testing-utils), [fms](https://github.com/foundation-model-stack/foundation-model-stack), [HF Transformers](https://huggingface.co/docs/hub/en/transformers), [torch](https://pytorch.org/get-started/locally/) and torch_sendnn. Depending on your use case, you may need other libraries as well.

For our example code, we will use the following libraries.
```python
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

Now, add the model setup and tokenizer details.
```python
# We will provide our model as a variant as below. If you have a model available locally, you can use model_path variable instead of variant.
variant = "ibm-granite/granite-3.0-8b-base" # or "ibm-ai-platform/micro-g3.3-8b-instruct-1b" etc.
model = get_model(
    architecture="hf_pretrained",
    variant=variant,
    device_type="cpu",
    data_type=torch.float16,
    fused_weights=False,
)
model.eval()
torch.set_grad_enabled(False)
model.compile(backend="sendnn") # Compile with the AIU sendnn backend

# Tokenize
tokenizer = AutoTokenizer.from_pretrained(variant)
```

Also, since we are using Granite decoder model, set compilation mode to offline_decoder instead of using the default.
```python
os.environ.setdefault("COMPILATION_MODE", "offline_decoder")
```

Now, let's define prompt.
```python
template = "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{}\n\n### Response:"
prompt = template.format("Provide a list of instructions for preparing chicken soup.")
input_ids = tokenizer.encode(prompt, return_tensors="pt")
input_ids, extra_generation_kwargs = pad_input_ids([input_ids.squeeze(0)], min_pad_length=math.ceil(input_ids.size(1)/64) * 64)
# only_last_token optimization
extra_generation_kwargs["last_n_tokens"] = 1
# Set a desired number
max_new_tokens = 16
```

That's it! We are ready to generate model response.
```python
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
```python
# Print output
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

### Response:
1. Gather ingredients: chicken, vegetables (carrots, celery, onions), herbs (parsley, thyme, bay leaves), salt, pep
...
```
You can find this code under the `run_granite3.py`. 