from fms.utils import tokenizers
import pytest
from fms.models import get_model
from fms.utils.generation import generate, pad_input_ids
import itertools
import torch
from aiu_fms_testing_utils.testing.validation import extract_validation_information, LogitsExtractorHook, GoldenTokenHook, capture_level_1_metrics, filter_failed_level_1_cases, load_validation_information, validate_level_0, top_k_loss_calculator
from aiu_fms_testing_utils.utils import warmup_model, sample_sharegpt_requests, ids_for_prompt
from aiu_fms_testing_utils.utils.aiu_setup import dprint
import os

model_dir = os.environ.get("FMS_TESTING_MODEL_DIR", "/tmp/models")
validation_info_dir = os.environ.get("FMS_TESTING_VALIDATION_INFO_DIR", "/tmp/models/validation_info")

LLAMA_194M = f"{model_dir}/llama-194m"
GRANITE_7B_BASE = f"{model_dir}/granite-7b-base"
GRANITE_8B_CODE_BASE = f"{model_dir}/granite-8b-code-base"
GRANITE_3_8B_CODE_BASE = f"{model_dir}/granite-3-8b-base"
SHARE_GPT_DATASET_PATH = os.environ.get("SHARE_GPT_DATASET_PATH","/tmp/devel/src/share_gpt.json")

LLAMA_194M_VALIDATION_INFO_PATH = os.environ.get("")

# pass custom model path list for eg: EXPORT FMS_TESTING_COMMON_MODEL_PATHS="/tmp/models/granite-3-8b-base,/tmp/models/granite-7b-base"
if os.environ.get("FMS_TESTING_COMMON_MODEL_PATHS") == None or os.environ.get("FMS_TESTING_COMMON_MODEL_PATHS") == "":
    common_model_paths = [LLAMA_194M, GRANITE_7B_BASE, GRANITE_8B_CODE_BASE, GRANITE_3_8B_CODE_BASE]
else:
    common_model_paths = os.environ.get("FMS_TESTING_COMMON_MODEL_PATHS").split(',')

# thresholds are chosen based on 1024 tokens per sequence
# 1% error threshold rate between cpu fp32 and cuda fp16
thresholds = (2.69946389,(-1.18985e-8, 1.26977e-8))
fail_thresholds = {
    LLAMA_194M: thresholds,
    GRANITE_7B_BASE: thresholds,
    GRANITE_8B_CODE_BASE: thresholds,
    GRANITE_3_8B_CODE_BASE: thresholds,
}

common_batch_sizes = [1, 2, 4, 8]
common_seq_lengths = [64, 2048]
common_max_new_tokens = [128]

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

def __find_eos_index(reference_tokens, eos_token_id, seq_length, max_new_tokens):
    result = []
    for sentence in reference_tokens:
        found_eos = False
        for token_idx, token in enumerate(sentence[seq_length:]):
            if token.item() == eos_token_id:
                found_eos = True
                result.append(token_idx)
                break
        if not found_eos:
            result.append(max_new_tokens)
    return result

def __filter_before_eos(l, filter_indexes):
    from itertools import groupby
    filtered_results = [list(g)[:filter_indexes[k]] for k, g in groupby(l, key=lambda x: x[0])]
    return [item for sublist in filtered_results for item in sublist]

def __load_validation_info(model_path, batch_size, seq_length, max_new_tokens, tokenizer, seed):
    path_to_validation_info = os.path.join(validation_info_dir, os.path.basename(model_path))
    validation_file_name = f"max_new_tokens-{max_new_tokens}_batch_size-{batch_size}_sequence_length-{seq_length}.validation_info_output.{seed}.out"
    full_path = os.path.join(path_to_validation_info, validation_file_name)

    if os.path.exists(full_path):
        return load_validation_information(full_path, "logits", batch_size, tokenizer)
    else:
        return None


@pytest.mark.parametrize("model_path,batch_size,seq_length,max_new_tokens", common_shapes)
def test_common_shapes(model_path, batch_size, seq_length, max_new_tokens):
    dprint(f"testing model={model_path}, batch_size={batch_size}, seq_length={seq_length}, max_new_tokens={max_new_tokens}")

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
    cpu_validation_info = __load_validation_info(model_path, batch_size, seq_length, max_new_tokens, tokenizer, 0)
    if cpu_validation_info is None:
        cpu_validation_info = extract_validation_information(
            validation_model,
            input_ids,
            max_new_tokens,
            LogitsExtractorHook(),
            attn_algorithm="math",
            **padding_kwargs
        )
    cpu_static_tokens = cpu_validation_info.get_info("tokens")
    eos_indexes = __find_eos_index(cpu_static_tokens, tokenizer.eos_token_id, seq_length, max_new_tokens)
    dprint("cpu validation info extracted for validation level 0 and validation level 1 (iter=0)")

    # first test validation level 0
    aiu_validation_info = extract_validation_information(
        model,
        input_ids,
        max_new_tokens,
        None,
        only_last_token=True,
        **padding_kwargs
    )
    dprint("aiu validation info extracted for validation level 0")

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
        total_tokens = 0
        for i in range(iters):

            if i != 0:
                input_ids, padding_kwargs = __prepare_inputs(batch_size, seq_length, tokenizer, seed=i)
                cpu_validation_info = __load_validation_info(model_path, batch_size, seq_length, max_new_tokens, tokenizer, i)
                if cpu_validation_info is None:
                    cpu_validation_info = extract_validation_information(
                        validation_model,
                        input_ids,
                        max_new_tokens,
                        LogitsExtractorHook(),
                        attn_algorithm="math",
                        **padding_kwargs
                    )
                cpu_static_tokens = cpu_validation_info.get_info("tokens")
                eos_indexes = __find_eos_index(cpu_static_tokens, tokenizer.eos_token_id, seq_length, max_new_tokens)
                dprint(f"cpu validation info extracted for validation level 1 - iter={i}")
            
            # generate aiu validation info
            aiu_validation_info = extract_validation_information(
                model,
                input_ids,
                max_new_tokens,
                GoldenTokenHook(cpu_static_tokens),
                only_last_token=True,
                **padding_kwargs
            )
            dprint(f"aiu validation info extracted for validation level 1 - iter={i}")
        
            level_1_metrics = capture_level_1_metrics(
                cpu_validation_info.get_info("logits"),
                aiu_validation_info.get_info("logits"),
                top_k_loss_calculator(20, _metric_calculator)
            )
            level_1_metrics = __filter_before_eos(level_1_metrics, eos_indexes)

            ce_threshold = fail_thresholds[model_path][0]
            diff_thresholds = fail_thresholds[model_path][1]

            ce_fail_responses = filter_failed_level_1_cases(level_1_metrics, lambda m: m[0] >= ce_threshold)
            diff_fail_responses = filter_failed_level_1_cases(level_1_metrics, lambda m: m[1] <= diff_thresholds[0] or m[1] >= diff_thresholds[1])

            ce_fail_responses_list.extend(ce_fail_responses)
            diff_fail_responses_list.extend(diff_fail_responses)
            total_tokens += len(level_1_metrics)
        
        diff_failure_rate = len(diff_fail_responses_list) / total_tokens
        assert diff_failure_rate < .02, f"failure rate for mean diff was too high: {diff_failure_rate}"
        ce_failure_rate = len(ce_fail_responses_list) / total_tokens
        assert ce_failure_rate < .03, f"failure rate for cross entropy loss was too high: {ce_failure_rate}"

        print("passed validation level 1")
    else:
        print("passed validation level 0")



