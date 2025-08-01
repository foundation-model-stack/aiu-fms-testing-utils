"""This module contains test related to compilation operation"""

# Standard
import os
import pytest
import time

# Third Party
from transformers.tokenization_utils_base import BatchEncoding
import torch

# First Party
from fms.models import get_model
from fms.utils import tokenizers
from fms.utils.generation import pad_input_ids

# Local
from aiu_fms_testing_utils.utils import (
    ids_for_prompt,
    get_env_to_datatype_list,
    sample_sharegpt_requests,
    warmup_model,
)
from aiu_fms_testing_utils.utils.aiu_setup import dprint

GRANITE_3p3_8B_INSTRUCT = "ibm-granite/granite-3.3-8b-instruct"
SHARE_GPT_DATASET_PATH = os.environ.get(
    "SHARE_GPT_DATASET_PATH", os.path.expanduser("~/share_gpt.json")
)

ATTN_TYPE = os.environ.get("FMS_TEST_SHAPES_ATTN_TYPE", "paged")
attention_map = {
    "sdpa": "sdpa_causal",
    "paged": "spyre_paged_attn",
    "math_fp8": "math_fp8",
    "paged_fp8": "spyre_paged_attn_fp8",
}
ATTN_NAME = attention_map[ATTN_TYPE]

COMPILE_DYNAMIC_SHAPE = os.environ.get("AIU_COMPILE_DYNAMIC_SHAPE", True)


common_model_paths = get_env_to_datatype_list("COMMON_MODEL_NAME", [GRANITE_3p3_8B_INSTRUCT])
common_batch_sizes = get_env_to_datatype_list("FMS_TEST_SHAPES_COMMON_BATCH_SIZES", [1])
common_seq_lengths = get_env_to_datatype_list("FMS_TEST_SHAPES_COMMON_SEQ_LENGTHS", [64])
common_max_new_tokens = get_env_to_datatype_list(
    "FMS_TEST_SHAPES_COMMON_MAX_NEW_TOKENS", [64]
)
common_expected_comp_time = get_env_to_datatype_list(
    "COMMON_COMPILATION_EXPECTED_TIME", [10]
)  # In minutes


if COMPILE_DYNAMIC_SHAPE:
    COMMON_SHAPE_TYPE = "dynamic"
else:
    COMMON_SHAPE_TYPE = "static"

COMMON_SHAPES = list(
    zip(
        common_model_paths,
        common_batch_sizes,
        common_seq_lengths,
        common_max_new_tokens,
        common_expected_comp_time,
    )
)


def __set_context_length(seq_len: int, max_new_tokens: int, batch_size: int) -> None:
    """
    This function sets the environment variables for maximum context length and batch size.

    It calculates the largest context by adding the sequence length and the maximum number of new tokens.

    Args:
        seq_len (int): The length of the input sequence.
        max_new_tokens (int): The maximum number of new tokens to generate.
        batch_size (int): The batch size for processing.

    This function sets the environment variables:
    - VLLM_DT_MAX_CONTEXT_LEN: The selected maximum context length.
    - VLLM_DT_MAX_BATCH_SIZE: The maximum batch size, with a minimum value of 2.
    """
    largest_context = seq_len + max_new_tokens
    os.environ["VLLM_DT_MAX_CONTEXT_LEN"] = str(largest_context)
    os.environ["VLLM_DT_MAX_BATCH_SIZE"] = str(max(batch_size, 2))

def __get_dummy_inputs(batch_size, seq_length, tokenizer):
    """
    This function creates dummy input tensors for a given sequence length.
    It uses the tokenizer to generate valid token IDs, excluding special
    tokens (beginning-of-sequence and end-of-sequence).

    Args:
        batch_size (int): The number of sequences in a batch.
        seq_length (int): The length of each sequence.
        tokenizer (Tokenizer): The tokenizer object used for tokenization.

    Returns:
        Tuple(input_ids, attention_masks)
            - input_ids (torch.Tensor): A tensor of shape (batch_size, seq_length)
              containing randomly sampled valid token IDs.
            - attention_masks (torch.Tensor): A tensor of shape (batch_size, seq_length)
              filled with ones, indicating that all tokens are attended to.
    """
    vocab_size = tokenizer.tokenizer.vocab_size
    bos_token_id = tokenizer.bos_token_id
    eos_token_id = tokenizer.eos_token_id
    special_token_ids = [bos_token_id, eos_token_id]
    # breakpoint()
    valid_token_ids = [
        i for i in range(1, vocab_size) if i not in set(special_token_ids)
    ]
    valid_token_ids_tensor = torch.tensor(valid_token_ids, dtype=torch.long)

    # Sample from the valid token ids
    input_ids = valid_token_ids_tensor[torch.randint(
            0, len(valid_token_ids_tensor), (batch_size, seq_length))]

    attention_masks = torch.ones((batch_size, seq_length))

    position_ids = torch.arange(seq_length, dtype=torch.long).unsqueeze(0).expand(batch_size, -1)

    # breakpoint()

    batch = BatchEncoding({
        "input_ids": input_ids,
        "mask": attention_masks,
        "position_ids": position_ids,
    })
    return batch



# TODO: Move this function outside for re-usability in other places
def __generate(model, input_ids, max_new_tokens, **kwargs):
    import torch_sendnn

    attention_specific_kwargs = {}
    attn_name = kwargs.get("attn_name", "sdpa")

    extra_kwargs = kwargs.get("extra_kwargs", {})

    if "paged" in attn_name:
        from aiu_fms_testing_utils.utils.paged import generate, adjust_inputs_to_batch
        input_ids, _extra_kwargs = adjust_inputs_to_batch(input_ids=input_ids, **extra_kwargs)
        extra_kwargs = {**_extra_kwargs, "attn_name": attn_name}
    else:
        # TODO: Add a unified generation dependent on attn_type
        from fms.utils.generation import generate

        attention_specific_kwargs["contiguous_cache"] = True
        attention_specific_kwargs["max_seq_len"] = input_ids.shape[1] + max_new_tokens
        extra_kwargs["only_last_token"] = True

    return generate(
        model,
        input_ids,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        extra_kwargs=extra_kwargs,
        **attention_specific_kwargs,
    )


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
    __set_context_length(seq_length, max_new_tokens, batch_size)

    dprint(
        f"testing model={model_path}, batch_size={batch_size}, seq_length={seq_length}, attn_type={ATTN_TYPE}",
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

    # prepare batch input
    batch = __get_dummy_inputs(batch_size, seq_length, tokenizer)
    # inputs, args = __prepare_inputs(batch_size, seq_length, tokenizer)
    # breakpoint()

    start_time = time.perf_counter()
    if COMMON_SHAPE_TYPE == "dynamic":
        COMPILE_DYNAMIC_SHAPE = True
    else:
        COMPILE_DYNAMIC_SHAPE = False

    model.compile(backend="sendnn", options={"sendnn.dynamic": COMPILE_DYNAMIC_SHAPE})
    warmup_model(
        model, batch["input_ids"], max_new_tokens, COMPILE_DYNAMIC_SHAPE, attn_name=ATTN_NAME, mask=batch["mask"], position_ids=batch["position_ids"]
    )
    extra_kwargs = {
        "position_ids": batch["position_ids"],
        "mask": batch["mask"],
        "attn_name": ATTN_NAME
    }
    __generate(model, batch["input_ids"], max_new_tokens=max_new_tokens, use_cache=True, extra_kwargs=extra_kwargs)

    end_time = time.perf_counter()

    assert (end_time - start_time) < expected_comp_time * 60