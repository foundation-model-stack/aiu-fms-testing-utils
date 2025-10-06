import math
import os
import torch

from aiu_fms_testing_utils.utils import warmup_model
from aiu_fms_testing_utils.utils.aiu_setup import dprint
from fms.models import get_model
from fms.utils.generation import generate, pad_input_ids
from torch_sendnn import torch_sendnn  # noqa: F401
from transformers import AutoTokenizer

# We will provide our model as a variant as below. If you have a model available locally, you can use model_path variable instead of variant.
variant = "ibm-granite/granite-3.0-8b-base"  # or "ibm-ai-platform/micro-g3.3-8b-instruct-1b" etc.
model = get_model(
    architecture="hf_pretrained",
    variant=variant,
    device_type="cpu",
    data_type=torch.float16,
    fused_weights=False,
)
model.eval()
torch.set_grad_enabled(False)
model.compile(backend="sendnn")  # Compile with the AIU sendnn backend

# Tokenize
tokenizer = AutoTokenizer.from_pretrained(variant)

os.environ.setdefault("COMPILATION_MODE", "offline_decoder")

template = "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{}\n\n### Response:"
prompt = template.format("Provide a list of instructions for preparing chicken soup.")
input_ids = tokenizer.encode(prompt, return_tensors="pt")
input_ids, extra_generation_kwargs = pad_input_ids(
    [input_ids.squeeze(0)], min_pad_length=math.ceil(input_ids.size(1) / 64) * 64
)
# only_last_token optimization
extra_generation_kwargs["last_n_tokens"] = 1
# Set a desired number
max_new_tokens = 16

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


# Print output
def print_result(result):
    output_str = tokenizer.convert_tokens_to_string(
        tokenizer.convert_ids_to_tokens(result)
    )
    dprint(output_str)
    print("...")


for i in range(result.shape[0]):
    print_result(result[i])
