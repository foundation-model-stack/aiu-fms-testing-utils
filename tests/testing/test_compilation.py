"""This module contains test related to compilation operation"""

# Standard
import os
import pytest
import time

# Third Party
import torch

# First Party
from fms.models import get_model
from fms.utils import tokenizers
from fms.utils.generation import pad_input_ids

# Local
from aiu_fms_testing_utils.utils import (
    ids_for_prompt,
    get_env_to_int_list,
    sample_sharegpt_requests,
    warmup_model,
)
from aiu_fms_testing_utils.utils.aiu_setup import dprint

GRANITE_3p3_8B_INSTRUCT = "ibm-granite/granite-3.3-8b-instruct"
SHARE_GPT_DATASET_PATH = os.environ.get(
    "SHARE_GPT_DATASET_PATH", os.path.expanduser("~/share_gpt.json")
)

ATTN_NAME = "spyre_paged_attn"

COMPILE_DYNAMIC_SHAPE = True


common_model_paths = get_env_to_int_list("COMMON_MODEL_NAME", [GRANITE_3p3_8B_INSTRUCT])
common_batch_sizes = get_env_to_int_list("FMS_TEST_SHAPES_COMMON_BATCH_SIZES", [1])
common_seq_lengths = get_env_to_int_list("FMS_TEST_SHAPES_COMMON_SEQ_LENGTHS", [64])
common_max_new_tokens = get_env_to_int_list(
    "FMS_TEST_SHAPES_COMMON_MAX_NEW_TOKENS", [64]
)
common_expected_comp_time = get_env_to_int_list(
    "COMMON_COMPILATION_EXPECTED_TIME", [10]
)  # In minutes

COMMON_SHAPE_TYPE = "dynamic"


if COMPILE_DYNAMIC_SHAPE:
    import bisect

    # the compiler supports certain max context lengths (VLLM_DT_MAX_CONTEXT_LEN)
    # this will ensure that we select smallest supported VLLM_DT_MAX_CONTEXT_LEN that fits the largest possible context (prompt size + max_new_tokens)
    __largest_context = max(common_seq_lengths) + max(common_max_new_tokens)
    __supported_context_lengths = [256, 512, 1024, 2048, 4096, 8192]
    os.environ["VLLM_DT_MAX_CONTEXT_LEN"] = str(
        __supported_context_lengths[
            bisect.bisect_left(__supported_context_lengths, __largest_context)
        ]
    )
    os.environ["VLLM_DT_MAX_BATCH_SIZE"] = str(max(max(common_batch_sizes), 2))

COMMON_SHAPES = list(
    zip(
        common_model_paths,
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
        seq_length // 2,
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
    if not COMPILE_DYNAMIC_SHAPE:
        torch.compiler.reset()
        torch._dynamo.reset()
        os.environ.pop("COMPILATION_MODE", None)


@pytest.mark.parametrize(
    "model_path,batch_size,seq_length,max_new_tokens,expected_comp_time", COMMON_SHAPES
)
def test_compilation_time(
    model_path, batch_size, seq_length, max_new_tokens, expected_comp_time
):
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
        data_type=torch.float16,
        fused_weights=False,
        **model_path_kwargs,
    )

    model.eval()
    torch.set_grad_enabled(False)

    # prepare input_ids
    input_ids, extra_kwargs = __prepare_inputs(batch_size, seq_length, tokenizer)
    extra_kwargs["attn_name"] = ATTN_NAME

    start_time = time.perf_counter()
    if COMMON_SHAPE_TYPE == "dynamic":
        COMPILE_DYNAMIC_SHAPE = True
    else:
        COMPILE_DYNAMIC_SHAPE = False

    model.compile(backend="sendnn", options={"sendnn.dynamic": COMPILE_DYNAMIC_SHAPE})
    warmup_model(
        model, input_ids, max_new_tokens, COMPILE_DYNAMIC_SHAPE, **extra_kwargs
    )
    end_time = time.perf_counter()

    assert (end_time - start_time) < expected_comp_time * 60