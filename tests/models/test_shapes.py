from fms.utils import tokenizers
import pytest
from fms.models import get_model
from fms.utils.generation import generate, pad_input_ids
import itertools
import torch
from aiu_fms_testing_utils.testing.validation import extract_validation_information, LogitsExtractorHook, GoldenTokenHook, capture_level_1_metrics, filter_failed_level_1_cases, validate_level_0, top_k_loss_calculator
from aiu_fms_testing_utils.utils import warmup_model, sample_sharegpt_requests, ids_for_prompt
import os

model_dir = os.environ.get("FMS_TESTING_MODEL_DIR", "/tmp/models")

LLAMA_194M = f"{model_dir}/llama-194m"
GRANITE_7B_BASE = f"{model_dir}/granite-7b-base"
GRANITE_8B_CODE_BASE = f"{model_dir}/granite-8b-code-base"
GRANITE_3_8B_CODE_BASE = f"{model_dir}/granite-3-8b-base"
SHARE_GPT_DATASET_PATH = os.environ.get("SHARE_GPT_DATASET_PATH","/tmp/devel/src/share_gpt.json")

# pass custom model path list for eg: EXPORT FMS_TESTING_COMMON_MODEL_PATHS="/tmp/models/granite-3-8b-base,/tmp/models/granite-7b-base"
if os.environ.get("FMS_TESTING_COMMON_MODEL_PATHS") == None or os.environ.get("FMS_TESTING_COMMON_MODEL_PATHS") == "":
    common_model_paths = [LLAMA_194M, GRANITE_7B_BASE, GRANITE_8B_CODE_BASE, GRANITE_3_8B_CODE_BASE]
else:
    common_model_paths = os.environ.get("FMS_TESTING_COMMON_MODEL_PATHS").split(',')

thresholds = (2.0,(-1.3e-8, 1.3e-8))
fail_thresholds = {
    LLAMA_194M: thresholds,
    GRANITE_7B_BASE: thresholds,
    GRANITE_8B_CODE_BASE: thresholds,
    GRANITE_3_8B_CODE_BASE: thresholds,
}

common_batch_sizes = [1, 2, 4, 8]
common_seq_lengths = [64, 2048]
common_max_new_tokens = [8, 128]

common_shapes = list(itertools.product(common_model_paths, common_batch_sizes, common_seq_lengths, common_max_new_tokens))

@pytest.fixture(autouse=True)
def reset_compiler():
    yield # run the test
    torch.compiler.reset()
    torch._dynamo.reset()

def __prepare_inputs(batch_size, seq_length, tokenizer, seed=0):
    prompts_and_sizes = sample_sharegpt_requests(SHARE_GPT_DATASET_PATH, batch_size, tokenizer, int(seq_length / 2), seq_length, seed)
    prompt_list = []
    for prompt, _ in prompts_and_sizes:
        prompt_list.append(ids_for_prompt(prompt, tokenizer))

    input_ids, padding_kwargs = pad_input_ids(prompt_list, min_pad_length=seq_length)
    return input_ids, padding_kwargs

@pytest.mark.parametrize("model_path,batch_size,seq_length,max_new_tokens", common_shapes)
def test_common_shapes(model_path, batch_size, seq_length, max_new_tokens):
    tokenizer = tokenizers.get_tokenizer(model_path)
    
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
    input_ids, padding_kwargs = __prepare_inputs(batch_size, seq_length, tokenizer)

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

        def _metric_calculator(r: torch.Tensor, t: torch.Tensor):
            cross_entropy = torch.nn.CrossEntropyLoss()(r, t.softmax(dim=1).to(dtype=torch.float32))
            diff = torch.mean(r.softmax(dim=1).to(dtype=torch.float32) - t.softmax(dim=1).to(dtype=torch.float32))
            return (cross_entropy, diff)

        iters = 1024 // max_new_tokens
        ce_fail_responses_list = []
        diff_fail_responses_list = []
        for i in range(iters):
            input_ids, padding_kwargs = __prepare_inputs(batch_size, seq_length, tokenizer, seed=i)
            
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
                aiu_validation_info.get_info("logits"),
                top_k_loss_calculator(20, _metric_calculator)
            )

            ce_threshold = fail_thresholds[model_path][0]
            diff_thresholds = fail_thresholds[model_path][1]

            ce_fail_responses = filter_failed_level_1_cases(level_1_metrics, lambda m: m[0] >= ce_threshold)
            diff_fail_responses = filter_failed_level_1_cases(level_1_metrics, lambda m: m[1] <= diff_thresholds[0] or m[1] >= diff_thresholds[1])

            ce_fail_responses_list.extend(ce_fail_responses)
            diff_fail_responses_list.extend(diff_fail_responses)

        # assert that the failure rate is less than double the 1% of cuda
        # ce_failure_rate = len(ce_fail_responses) / len(level_1_metrics)
        # assert ce_failure_rate < 0.02, f"failure rate for cross-entropy loss was too high: {ce_failure_rate}"
        
        diff_failure_rate = len(diff_fail_responses_list) / (1024 * batch_size)
        assert diff_failure_rate < 0.02, f"failure rate for mean diff was too high: {diff_failure_rate}"
        ce_failure_rate = len(ce_fail_responses_list) / (1024 * batch_size)
        print(f"failure rate for cross entropy loss was too high: {ce_failure_rate}")
        
        print("passed validation level 1")
    else:
        print("passed validation level 0")



