"""This module contains test related to compilation operation"""

# Standard
import itertools
import os
import pytest
import time

# Third Party
from torch import distributed as dist
import torch

# First Party
from fms.models import get_model
from fms.utils import generation, tokenizers
from fms.utils.generation import pad_input_ids

# Local
from aiu_fms_testing_utils.utils import ids_for_prompt, sample_sharegpt_requests, warmup_model
from aiu_fms_testing_utils.utils.aiu_setup import dprint

GRANITE_3p3_8B_INSTRUCT = "ibm-granite/granite-3.3-8b-instruct"
SHARE_GPT_DATASET_PATH = os.environ.get(
    "SHARE_GPT_DATASET_PATH", os.path.expanduser("~/share_gpt.json")
)

ATTN_NAME = "spyre_paged_attn"

compile_dynamic_sendnn = True

common_model_paths = [GRANITE_3p3_8B_INSTRUCT]
common_batch_sizes = [1]
common_seq_lengths = [256]
common_shape_types = ["dynamic"]
common_max_new_tokens = [128]
common_expected_comp_time = [10] # In minutes

if compile_dynamic_sendnn:
    os.environ["VLLM_DT_MAX_CONTEXT_LEN"] = str(
        (((max(common_seq_lengths) + max(common_max_new_tokens)) // 64) + 1) * 64
    )
    os.environ["VLLM_DT_MAX_BATCH_SIZE"] = str(max(max(common_batch_sizes), 2))

common_shapes = list(
    zip(
        common_model_paths,
        common_shape_types,
        common_batch_sizes,
        common_seq_lengths,
        common_max_new_tokens,
        common_expected_comp_time,
    )
)


# TODO: This is copied from test_decoders.py would be good to consolidate
def __prepare_inputs(batch_size, seq_length, tokenizer, seed=0):
    prompts_and_sizes = sample_sharegpt_requests(
        SHARE_GPT_DATASET_PATH,
        batch_size,
        tokenizer,
        int(seq_length / 2),
        seq_length,
        seed,
    )
    prompt_list = []
    for prompt, _ in prompts_and_sizes:
        prompt_list.append(ids_for_prompt(prompt, tokenizer))

    input_ids, extra_kwargs = pad_input_ids(prompt_list, min_pad_length=seq_length)
    return input_ids, extra_kwargs


@pytest.fixture(autouse=True)
def reset_compiler():
    yield  # run the test
    if not compile_dynamic_sendnn:
        torch.compiler.reset()
        torch._dynamo.reset()
        os.environ.pop("COMPILATION_MODE", None)


@pytest.mark.parametrize(
    "model_path,shape_type,batch_size,seq_length,max_new_tokens,expected_comp_time", common_shapes
)
def test_compilation_time(model_path, shape_type, batch_size, seq_length, max_new_tokens, expected_comp_time):
    """Test to validate time taken for model compilation."""
    torch.manual_seed(42)
    torch.set_default_dtype(torch.float16)
    os.environ["COMPILATION_MODE"] = "offline_decoder"

    dprint(
        f"testing model={model_path}, batch_size={batch_size}, seq_length={seq_length}"
    )

    if os.path.exists(model_path):
        model_path_kwargs = {"model_path": model_path}
    else:
        model_path_kwargs = {"variant": model_path}

    tokenizer = tokenizers.get_tokenizer(model_path)

    # prepare the AIU model
    model = get_model(
        architecture="hf_pretrained",
        device_type="cpu",
        data_type= torch.float16,
        fused_weights=False,
        **model_path_kwargs,
    )

    model.eval()
    torch.set_grad_enabled(False)

    # prepare input_ids
    input_ids, extra_kwargs = __prepare_inputs(batch_size, seq_length, tokenizer)
    extra_kwargs["attn_name"] = ATTN_NAME

    start_time = time.perf_counter()
    if shape_type == "dynamic":
        compile_dynamic_sendnn = True
    else:
        compile_dynamic_sendnn = False

    model.compile(
        backend="sendnn", options={"sendnn.dynamic": compile_dynamic_sendnn}
    )
    warmup_model(
        model,
        input_ids,
        max_new_tokens,
        compile_dynamic_sendnn,
        use_cache=False,
        **extra_kwargs
    )
    end_time = time.perf_counter()

    assert (end_time - start_time) < expected_comp_time * 60