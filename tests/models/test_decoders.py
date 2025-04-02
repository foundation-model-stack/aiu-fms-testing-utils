from fms.utils import tokenizers
import pytest
from fms.models import get_model
from fms.utils.generation import pad_input_ids
import itertools
import torch
from aiu_fms_testing_utils.testing.validation import extract_validation_information, LogitsExtractorHook, GoldenTokenHook, capture_level_1_metrics, filter_failed_level_1_cases, load_validation_information, validate_level_0, top_k_loss_calculator
from aiu_fms_testing_utils.utils import warmup_model, sample_sharegpt_requests, ids_for_prompt, _prepare_model_inputs_hook
from aiu_fms_testing_utils.utils.aiu_setup import dprint
import os

ORIGINAL_HF_HOME = os.environ.get("HF_HOME", None)

# Add models to test here
LLAMA_3p1_8B_INSTRUCT = "meta-llama/Llama-3.1-8B-Instruct"
GRANITE_3p2_8B_INSTRUCT = "ibm-granite/granite-3.2-8b-instruct"

SHARE_GPT_DATASET_PATH = os.environ.get("SHARE_GPT_DATASET_PATH", os.path.expanduser("~/share_gpt.json"))
USE_MICRO_MODELS = os.environ.get("FMS_TEST_SHAPES_USE_MICRO_MODELS", "1") == "1"
validation_info_dir = os.environ.get("FMS_TEST_SHAPES_VALIDATION_INFO_DIR", "/tmp/models/validation_info")
common_model_paths = os.environ.get("FMS_TEST_SHAPES_COMMON_MODEL_PATHS", [LLAMA_3p1_8B_INSTRUCT, GRANITE_3p2_8B_INSTRUCT])
# for validation level 1, the default is a failure rate of 1%
# set this environment variable if you would like to relax that threshold
failure_rate_threshold = os.environ.get("FMS_TEST_SHAPES_FAILURE_THRESHOLD", 0.01)
default_metrics_threshold = os.environ.get("FMS_TEST_SHAPES_METRICS_THRESHOLD",(2.0, (-1.0e-8,1.0e-8)))
save_validation_info_outputs = os.environ.get("FMS_TEST_SHAPES_SAVE_VALIDATION_INFO_OUTPUTS", "0") == "1"
common_batch_sizes = os.environ.get("FMS_TEST_SHAPES_COMMON_BATCH_SIZES", [1, 2, 4, 8])
common_seq_lengths = os.environ.get("FMS_TEST_SHAPES_COMMON_SEQ_LENGTHS", [64, 2048])
common_max_new_tokens = os.environ.get("FMS_TEST_SHAPES_COMMON_MAX_NEW_TOKENS", [128])

if USE_MICRO_MODELS:
    validation_info_dir = os.path.join(validation_info_dir, "tiny_models")

# pass custom model path list for eg: EXPORT FMS_TESTING_COMMON_MODEL_PATHS="/tmp/models/granite-3-8b-base,/tmp/models/granite-7b-base"
if isinstance(common_model_paths, str):
    common_model_paths = common_model_paths.split(",")

# pass custom failure rate threshold as float
if isinstance(failure_rate_threshold, str):
    failure_rate_threshold = float(failure_rate_threshold)

# pass custom default metrics threshold as a comma separated str of floats <cross-entropy threshold>,<mean diff min threshold>,<mean diff max threshold>
if isinstance(default_metrics_threshold, str):
    m_list = [float(m) for m in default_metrics_threshold.split(",")]
    default_metrics_threshold = (m_list[0], (m_list[1], m_list[2]))

# pass custom common batch sizes as a comma separated str of ints
if isinstance(common_batch_sizes, str):
    common_batch_sizes = [int(bs) for bs in common_batch_sizes.split(",")]

# pass custom common seq lengths as a comma separated str of ints
if isinstance(common_seq_lengths, str):
    common_seq_lengths = [int(sl) for sl in common_seq_lengths.split(",")]

# pass custom common max new tokens as a comma separated str of ints
if isinstance(common_max_new_tokens, str):
    common_max_new_tokens = [int(mnt) for mnt in common_max_new_tokens.split(",")]

common_shapes = list(itertools.product(common_model_paths, common_batch_sizes, common_seq_lengths, common_max_new_tokens))

# thresholds are chosen based on 1024 tokens per sequence
# 1% error threshold rate between cpu fp32 and cuda fp16
# if a models failure thresholds do not exist in this dict, default to the default_metrics_threshold defined above
fail_thresholds = {
    LLAMA_3p1_8B_INSTRUCT: (3.7392955756187423, (-1.0430812658057675e-08, 1.0401941685778344e-08)),
    GRANITE_3p2_8B_INSTRUCT: (2.996668996810913, (-8.911825961632757e-09, 8.75443184611413e-09)),
}

@pytest.fixture(autouse=True)
def reset_compiler():
    yield # run the test
    torch.compiler.reset()
    torch._dynamo.reset()
    os.environ.pop('COMPILATION_MODE', None)
    if ORIGINAL_HF_HOME is None:
        os.environ.pop('HF_HOME', None)
    else:
        os.environ['HF_HOME'] = ORIGINAL_HF_HOME

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

def __get_validation_info_full_path(model_path, batch_size, seq_length, max_new_tokens, seed):
    validation_file_name = f"{model_path.replace('/', '--')}_max-new-tokens-{max_new_tokens}_batch-size-{batch_size}_seq-length-{seq_length}_dtype-fp16.cpu_validation_info.{seed}.out"
    full_path = os.path.join(validation_info_dir, validation_file_name)
    return full_path

def __load_validation_info(model_path, batch_size, seq_length, max_new_tokens, tokenizer, seed):
    full_path = __get_validation_info_full_path(model_path, batch_size, seq_length, max_new_tokens, seed)
    
    if os.path.exists(full_path):
        return load_validation_information(full_path, "logits", batch_size, tokenizer)
    else:
        return None

@pytest.mark.parametrize("model_path,batch_size,seq_length,max_new_tokens", common_shapes)
def test_common_shapes(model_path, batch_size, seq_length, max_new_tokens):
    os.environ["COMPILATION_MODE"] = "offline_decoder"
    
    if "HF_HOME" not in os.environ:
        os.environ["HF_HOME"] = "/tmp/models/hf_cache"

    dprint(f"testing model={model_path}, batch_size={batch_size}, seq_length={seq_length}, max_new_tokens={max_new_tokens}, micro_model={USE_MICRO_MODELS}")

    if USE_MICRO_MODELS:
        micro_model_kwargs = {"architecture": "hf_configured", "nlayers": 3}
    else:
        micro_model_kwargs  = {"architecture": "hf_pretrained"}
    
    if not USE_MICRO_MODELS and os.path.exists(model_path):
        model_path_kwargs = {"model_path": model_path}
    else:
        model_path_kwargs = {"variant": model_path}
    
    get_model_kwargs = {**model_path_kwargs, **micro_model_kwargs}

    tokenizer = tokenizers.get_tokenizer(model_path)
    
    # prepare the AIU model
    model = get_model(
        device_type="cpu",
        fused_weights=False,
        **get_model_kwargs
    )

    model.eval()
    torch.set_grad_enabled(False)
    model.compile(backend="sendnn_decoder")

    # prepare the cpu model
    validation_model = get_model(
        device_type="cpu",
        data_type=torch.float32,
        fused_weights=False,
        **get_model_kwargs
    )

    if USE_MICRO_MODELS:
        validation_model.load_state_dict(model.state_dict())

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
    
    if save_validation_info_outputs:
        cpu_validation_info.save(__get_validation_info_full_path(model_path, batch_size, seq_length, max_new_tokens, 0))
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

    # validate level 0
    failed_responses = validate_level_0(aiu_validation_info.get_info("tokens"), cpu_static_tokens)

    # if level 0 fails validation, validate level 1
    if len(failed_responses) != 0:
        print("failed validation level 0, testing validation level 1")

        # metric calculator based on the cross-entropy and mean diff for each decode step
        def _metric_calculator(r: torch.Tensor, t: torch.Tensor):
            cross_entropy = torch.nn.CrossEntropyLoss()(r, t.softmax(dim=1).to(dtype=torch.float32))
            diff = torch.mean(r.softmax(dim=1).to(dtype=torch.float32) - t.softmax(dim=1).to(dtype=torch.float32))
            return (cross_entropy, diff)

        iters = 1024 // max_new_tokens
        ce_fail_responses_list = []
        diff_fail_responses_list = []
        total_tokens = 0
        for i in range(iters):

            # for iteration 0, we have computed the cpu validation info in the prior step for seed=0, so skip
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
                dprint(f"cpu validation info extracted for validation level 1 - iter={i}")
                if save_validation_info_outputs:
                    cpu_validation_info.save(__get_validation_info_full_path(model_path, batch_size, seq_length, max_new_tokens, i))
                cpu_static_tokens = cpu_validation_info.get_info("tokens")
                eos_indexes = __find_eos_index(cpu_static_tokens, tokenizer.eos_token_id, seq_length, max_new_tokens)
            
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
        
            # capture all level 1 metrics
            level_1_metrics = capture_level_1_metrics(
                cpu_validation_info.get_info("logits"),
                aiu_validation_info.get_info("logits"),
                top_k_loss_calculator(20, _metric_calculator)
            )
            # only consider those metrics captured prior to the eos
            level_1_metrics = __filter_before_eos(level_1_metrics, eos_indexes)

            ce_threshold, diff_thresholds = fail_thresholds.get(model_path, default_metrics_threshold)

            # get all failed responses for each metric
            ce_fail_responses = filter_failed_level_1_cases(level_1_metrics, lambda m: m[0] >= ce_threshold)
            diff_fail_responses = filter_failed_level_1_cases(level_1_metrics, lambda m: m[1] <= diff_thresholds[0] or m[1] >= diff_thresholds[1])

            ce_fail_responses_list.extend(ce_fail_responses)
            diff_fail_responses_list.extend(diff_fail_responses)
            total_tokens += len(level_1_metrics)
        
        # test the failure rates for across all tokens
        diff_failure_rate = len(diff_fail_responses_list) / total_tokens
        assert diff_failure_rate < failure_rate_threshold, f"failure rate for mean diff was too high: {diff_failure_rate}"
        ce_failure_rate = len(ce_fail_responses_list) / total_tokens
        assert ce_failure_rate < failure_rate_threshold, f"failure rate for cross entropy loss was too high: {ce_failure_rate}"

        print("passed validation level 1")
    else:
        print("passed validation level 0")

@pytest.mark.parametrize("use_static_shapes", [False, True], ids=["dynamic_shapes", "static_shapes"])
def test_warmup_multiple_shapes(use_static_shapes):
    shapes = [
        (1, 64, 4),
        (2, 64, 4),
        (1, 128, 8),
    ]

    tokenizer = tokenizers.get_tokenizer(GRANITE_3p2_8B_INSTRUCT)

    reference_model = get_model(
        architecture="hf_configured",
        variant=GRANITE_3p2_8B_INSTRUCT,
        device_type="cpu",
        fused_weights=False,
        nlayers=3
    )

    model = get_model(
        architecture="hf_configured",
        variant=GRANITE_3p2_8B_INSTRUCT,
        device_type="cpu",
        fused_weights=False,
        nlayers=3
    )

    reference_model.load_state_dict(model.state_dict())

    model.eval()
    reference_model.eval()

    torch.set_grad_enabled(False)
    model.compile(backend="sendnn_decoder")

    if use_static_shapes:
        ctx = torch._dynamo.config.patch(
            assume_static_by_default=True,
            dynamic_shapes=False,
            automatic_dynamic_shapes=False,
            cache_size_limit=1000,
        )
    else:
        ctx = torch._dynamo.config.patch(
            dynamic_shapes=True,
        )

    # metric calculator based on the cross-entropy and mean diff for each decode step
    def _metric_calculator(r: torch.Tensor, t: torch.Tensor):
        cross_entropy = torch.nn.CrossEntropyLoss()(r, t.softmax(dim=1).to(dtype=torch.float32))
        diff = torch.mean(r.softmax(dim=1).to(dtype=torch.float32) - t.softmax(dim=1).to(dtype=torch.float32))
        return (cross_entropy, diff)
    
    with ctx:

        for bs, sl, mnt in shapes:
            # prepare input_ids
            prompt_list = []
            for i in range(bs):
                prompt_list.append(torch.randint(0, model.config.src_vocab_size, (sl - 2 * i,), dtype=torch.long))

            input_ids, padding_kwargs = pad_input_ids(prompt_list, min_pad_length=sl)
            # warmup aiu model
            warmup_model(model, input_ids, mnt, **padding_kwargs)
    
        # perform 3 inference, making sure ordering does not affect things
        for _ in range(3):
            shapes.reverse()
            for bs, sl, mnt in shapes:
                prompt_list = []
                for i in range(bs):
                    prompt_list.append(torch.randint(0, model.config.src_vocab_size, (sl - 2 * i,), dtype=torch.long))
                input_ids, padding_kwargs = pad_input_ids(prompt_list, min_pad_length=sl)

                cpu_validation_info = extract_validation_information(
                    reference_model,
                    input_ids,
                    mnt,
                    LogitsExtractorHook(),
                    attn_algorithm="math",
                    **padding_kwargs
                )
                cpu_static_tokens = cpu_validation_info.get_info("tokens")
                eos_indexes = __find_eos_index(cpu_static_tokens, tokenizer.eos_token_id, sl, mnt)

                aiu_validation_info = extract_validation_information(
                    model,
                    input_ids,
                    mnt,
                    GoldenTokenHook(cpu_static_tokens),
                    only_last_token=True,
                    **padding_kwargs
                )

                level_1_metrics = capture_level_1_metrics(
                    cpu_validation_info.get_info("logits"),
                    aiu_validation_info.get_info("logits"),
                    top_k_loss_calculator(20, _metric_calculator)
                )

                # only consider those metrics captured prior to the eos
                level_1_metrics = __filter_before_eos(level_1_metrics, eos_indexes)

                ce_threshold, diff_thresholds = fail_thresholds.get(GRANITE_3p2_8B_INSTRUCT, default_metrics_threshold)

                # get all failed responses for each metric
                ce_fail_responses = filter_failed_level_1_cases(level_1_metrics, lambda m: m[0] >= ce_threshold)
                diff_fail_responses = filter_failed_level_1_cases(level_1_metrics, lambda m: m[1] <= diff_thresholds[0] or m[1] >= diff_thresholds[1])
                assert len(ce_fail_responses) == 0
                assert len(diff_fail_responses) == 0
