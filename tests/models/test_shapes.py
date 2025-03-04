import pytest
from fms.models import get_model
from fms.utils.generation import generate, pad_input_ids
import itertools
import torch
from aiu_fms_testing_utils.testing.validation import extract_validation_information, LogitsExtractorHook, GoldenTokenHook, capture_level_1_metrics, filter_failed_level_1_cases, validate_level_0
from aiu_fms_testing_utils.utils import warmup_model, aiu_setup
from torch_sendnn import backends
import os

model_dir = os.environ.get("FMS_TESTING_MODEL_DIR", "/tmp/models")

LLAMA_194M = f"{model_dir}/llama-194m"
GRANITE_7B_BASE = f"{model_dir}/granite-7b-base"
GRANITE_8B_CODE_BASE = f"{model_dir}/granite-8b-code-base"
GRANITE_3_8B_CODE_BASE = f"{model_dir}/granite-3-8b-base"

# pass custom model path list for eg: EXPORT FMS_TESTING_COMMON_MODEL_PATHS="/tmp/models/granite-3-8b-base,/tmp/models/granite-7b-base"
if os.environ.get("FMS_TESTING_COMMON_MODEL_PATHS") == None or os.environ.get("FMS_TESTING_COMMON_MODEL_PATHS") == "":
    common_model_paths = [LLAMA_194M, GRANITE_7B_BASE, GRANITE_8B_CODE_BASE, GRANITE_3_8B_CODE_BASE]
else:
    common_model_paths = os.environ.get("FMS_TESTING_COMMON_MODEL_PATHS").split(',')
    
common_batch_sizes = [1, 2, 4, 8]
common_seq_lengths = [64, 2048]
common_max_new_tokens = [8, 128]

common_shapes = list(itertools.product(common_model_paths, common_batch_sizes, common_seq_lengths, common_max_new_tokens))

@pytest.fixture(autouse=True)
def reset_compiler():
    yield # run the test
    torch.compiler.reset()
    torch._dynamo.reset()

@pytest.mark.parametrize("model_path,batch_size,seq_length,max_new_tokens", common_shapes)
def test_common_shapes(model_path, batch_size, seq_length, max_new_tokens):
    # prepare the AIU model
    model = get_model(
        "hf_pretrained",
        model_path=model_path,
        device_type="cpu",
        fused_weights=False,
    )

    model.eval()
    torch.set_grad_enabled(False)
    model.compile(backend="sendnn_decoder")

    # prepare the cpu model
    validation_model = get_model(
        "hf_pretrained",
        model_path=model_path,
        device_type="cpu",
    )

    # prepare input_ids
    prompt_list = []
    for i in range(batch_size):
        prompt_list.append(torch.randint(0, model.config.src_vocab_size, (seq_length - 2 * i,), dtype=torch.long))

    input_ids, padding_kwargs = pad_input_ids(prompt_list, min_pad_length=seq_length)

    # warmup aiu model
    warmup_model(model, input_ids, max_new_tokens, **padding_kwargs)

    # generate cpu validation info
    cpu_validation_info = extract_validation_information(
        validation_model,
        input_ids,
        max_new_tokens,
        LogitsExtractorHook(),
        attn_algorithm="math",
        **padding_kwargs
    )
    cpu_static_tokens = cpu_validation_info.get_info("tokens")

    # first test validation level 0
    aiu_validation_info = extract_validation_information(
        model,
        input_ids,
        max_new_tokens,
        None,
        only_last_token=True,
        **padding_kwargs
    )

    failed_responses = validate_level_0(aiu_validation_info.get_info("tokens"), cpu_static_tokens)

    if len(failed_responses) != 0:
        print("failed validation level 0, testing validation level 1")
        # generate aiu validation info
        aiu_validation_info = extract_validation_information(
            model,
            input_ids,
            max_new_tokens,
            GoldenTokenHook(cpu_static_tokens),
            only_last_token=True,
            **padding_kwargs
        )
        
        level_1_metrics = capture_level_1_metrics(
            cpu_validation_info.get_info("logits"),
            aiu_validation_info.get_info("logits")
        )

        failed_responses = filter_failed_level_1_cases(level_1_metrics, lambda m: m >= 64.0)
        assert len(failed_responses) == 0
        print("passed validation level 1")
    else:
        print("passed validation level 0")



