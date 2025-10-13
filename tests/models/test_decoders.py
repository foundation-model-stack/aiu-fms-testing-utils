from fms.models.hf.utils import AutoConfig
from fms.utils import serialization
import pytest
from fms.models import get_model
from fms.utils.generation import pad_input_ids
import itertools
import torch
from torch import distributed as dist
from torch.fx.experimental import _config as fx_config
from torch_sendnn.backends.sendnn_backend import _get_global_state
from torch_sendnn.utils.graph_cache import SpyreGraphCache

from aiu_fms_testing_utils.testing.validation import (
    extract_validation_information,
    LogitsExtractorHook,
    GoldenTokenHook,
    capture_level_1_metrics,
    filter_failed_level_1_cases,
    get_validation_info_path,
    load_validation_information,
    validate_level_0,
    top_k_loss_calculator,
    find_validation_info_path,
)
from aiu_fms_testing_utils.utils import (
    warmup_model,
    sample_sharegpt_requests,
)
from aiu_fms_testing_utils.utils.paged import KVCACHE_NUM_BLOCKS_HINT
import json
from transformers import AutoTokenizer

from aiu_fms_testing_utils.utils.aiu_setup import dprint, aiu_dist_setup
import shutil
import os

try:
    from fms_mo.aiu_addons.gptq import gptq_aiu_adapter, gptq_aiu_linear  # noqa: F401

    GPTQ_ENABLED = True
except ImportError:
    GPTQ_ENABLED = False

MICRO_MODELS_HOME = os.environ.get(
    "FMS_TEST_SHAPES_MICRO_MODELS_HOME", "/mnt/home/models/tiny-models"
)

# Add models to test here
LLAMA_3p1_8B_INSTRUCT = "meta-llama/Llama-3.1-8B-Instruct"
GRANITE_3p2_8B_INSTRUCT = "ibm-granite/granite-3.2-8b-instruct"
GRANITE_3p3_8B_INSTRUCT = "ibm-granite/granite-3.3-8b-instruct"
GRANITE_20B_CODE_INSTRUCT_8K = "ibm-granite/granite-20b-code-instruct-8k"
LLAMA_3p1_70B_INSTRUCT = "meta-llama/Llama-3.1-70B-Instruct"

MICRO_MODEL_MAPPING = {
    LLAMA_3p1_8B_INSTRUCT: os.path.join(
        MICRO_MODELS_HOME, "llama-3.1-8b-layers-3-step-24000"
    ),
    GRANITE_3p2_8B_INSTRUCT: os.path.join(
        MICRO_MODELS_HOME, "granite-3.2-8b-layers-3-step-100000"
    ),
    # FIXME: Because this uses the same config as 3.2, re-using here, but should update
    GRANITE_3p3_8B_INSTRUCT: os.path.join(
        MICRO_MODELS_HOME, "granite-3.3-8b-layers-3-step-100000"
    ),
    LLAMA_3p1_70B_INSTRUCT: os.path.join(
        MICRO_MODELS_HOME, "llama-3.1-70b-layers-3-step-24000"
    ),
}

SHARE_GPT_DATASET_PATH = os.environ.get(
    "SHARE_GPT_DATASET_PATH", os.path.expanduser("~/share_gpt.json")
)
USE_MICRO_MODELS = os.environ.get("FMS_TEST_SHAPES_USE_MICRO_MODELS", "1") == "1"
USE_DISTRIBUTED = os.environ.get("FMS_TEST_SHAPES_DISTRIBUTED", "0") == "1"
TIMING = os.environ.get("TIMING", "")
CUMULATIVE_TEST_TOKENS_PER_SEQUENCE = int(
    os.environ.get("FMS_TEST_SHAPES_CUMULATIVE_TEST_TOKENS_PER_SEQUENCE", "1024")
)
ATTN_TYPE = os.environ.get("FMS_TEST_SHAPES_ATTN_TYPE", "sdpa")
ATTENTION_MAP = {
    "sdpa": "sdpa_causal",
    "paged": "spyre_paged_attn",
    "math_fp8": "math_fp8",
    "paged_fp8": "spyre_paged_attn_fp8",
}
ATTN_NAME = ATTENTION_MAP[ATTN_TYPE]

CPU_DTYPE = "fp8" if "fp8" in ATTN_TYPE else "fp32"

FORCE_VALIDATION_LEVEL_1 = (
    os.environ.get("FMS_TEST_SHAPES_FORCE_VALIDATION_LEVEL_1", "0") == "1"
)
SKIP_ASSERTIONS = os.environ.get("FMS_TEST_SHAPES_SKIP_ASSERTIONS", {})
VALIDATION_INFO_DIR = os.environ.get(
    "FMS_TEST_SHAPES_VALIDATION_INFO_DIR", "/tmp/models/validation_info"
)
COMMON_MODEL_PATHS = os.environ.get(
    "FMS_TEST_SHAPES_COMMON_MODEL_PATHS",
    [
        LLAMA_3p1_8B_INSTRUCT,
        GRANITE_3p2_8B_INSTRUCT,
        GRANITE_3p3_8B_INSTRUCT,
        GRANITE_20B_CODE_INSTRUCT_8K,
        LLAMA_3p1_70B_INSTRUCT,
    ],
)
MODEL_CONFIGURATION_PATH = os.environ.get(
    "FMS_TEST_SHAPES_FROM_MODEL_CONFIGURATION", ""
)
MODEL_CONFIGURATION_FREQUENCY = os.environ.get(
    "FMS_TEST_SHAPES_FROM_MODEL_CONFIGURATION_FREQUENCY", "0"
)

# for validation level 1, the default is a failure rate of 1%
# set this environment variable if you would like to relax that threshold
FAILURE_RATE_THRESHOLD = os.environ.get("FMS_TEST_SHAPES_FAILURE_THRESHOLD", 0.01)
DEFAULT_METRICS_THRESHOLD = os.environ.get(
    "FMS_TEST_SHAPES_METRICS_THRESHOLD", (3.0, 0.001)
)
SAVE_VALIDATION_INFO_OUTPUTS = (
    os.environ.get("FMS_TEST_SHAPES_SAVE_VALIDATION_INFO_OUTPUTS", "0") == "1"
)
COMMON_BATCH_SIZES = os.environ.get("FMS_TEST_SHAPES_COMMON_BATCH_SIZES", [1, 2, 4, 8])
COMMON_SEQ_LENGTHS = os.environ.get("FMS_TEST_SHAPES_COMMON_SEQ_LENGTHS", [64, 2048])
COMMON_MAX_NEW_TOKENS = os.environ.get("FMS_TEST_SHAPES_COMMON_MAX_NEW_TOKENS", [128])

if USE_DISTRIBUTED:
    dist.init_process_group()
    aiu_dist_setup(dist.get_rank(), dist.get_world_size())
    SAVE_VALIDATION_INFO_OUTPUTS = SAVE_VALIDATION_INFO_OUTPUTS and (
        dist.get_rank() == 0
    )

if USE_MICRO_MODELS:
    VALIDATION_INFO_DIR = os.path.join(VALIDATION_INFO_DIR, "tiny_models")

# pass custom model path list for eg: EXPORT FMS_TESTING_COMMON_MODEL_PATHS="/tmp/models/granite-3-8b-base,/tmp/models/granite-7b-base"
if isinstance(COMMON_MODEL_PATHS, str):
    COMMON_MODEL_PATHS = COMMON_MODEL_PATHS.split(",")

# pass custom failure rate threshold as float
if isinstance(FAILURE_RATE_THRESHOLD, str):
    FAILURE_RATE_THRESHOLD = float(FAILURE_RATE_THRESHOLD)

# pass custom default metrics threshold as a comma separated str of floats <cross-entropy threshold>,<mean diff threshold>
if isinstance(DEFAULT_METRICS_THRESHOLD, str):
    DEFAULT_METRICS_THRESHOLD = tuple(
        [float(m) for m in DEFAULT_METRICS_THRESHOLD.split(",")]
    )

# pass custom common batch sizes as a comma separated str of ints
if isinstance(COMMON_BATCH_SIZES, str):
    COMMON_BATCH_SIZES = [int(bs) for bs in COMMON_BATCH_SIZES.split(",")]

# pass custom common seq lengths as a comma separated str of ints
if isinstance(COMMON_SEQ_LENGTHS, str):
    COMMON_SEQ_LENGTHS = [int(sl) for sl in COMMON_SEQ_LENGTHS.split(",")]

# pass custom common max new tokens as a comma separated str of ints
if isinstance(COMMON_MAX_NEW_TOKENS, str):
    COMMON_MAX_NEW_TOKENS = [int(mnt) for mnt in COMMON_MAX_NEW_TOKENS.split(",")]

# pass metrics to skip as a comma separated list (ce,mean_diff)
if isinstance(SKIP_ASSERTIONS, str):
    _skip_assertions = []
    for metric in SKIP_ASSERTIONS.split(","):
        metric = metric.lower()
        if metric not in {"ce", "mean_diff"}:
            pytest.fail(
                "FMS_TEST_SHAPES_SKIP_ASSERTIONS can only accept metrics ce and mean_diff"
            )
        _skip_assertions.append(metric)
    SKIP_ASSERTIONS = set(_skip_assertions)

COMPILE_DYNAMIC_SENDNN = ATTN_TYPE == "paged"

if COMPILE_DYNAMIC_SENDNN:
    import bisect

    # the compiler supports certain max context lengths (VLLM_DT_MAX_CONTEXT_LEN)
    # this will ensure that we select smallest supported VLLM_DT_MAX_CONTEXT_LEN that fits the largest possible context (prompt size + max_new_tokens)
    __largest_context = max(COMMON_SEQ_LENGTHS) + max(COMMON_MAX_NEW_TOKENS)
    __supported_context_lengths = [256, 512, 1024, 2048, 4096, 8192, 16384, 32768]
    os.environ["VLLM_DT_MAX_CONTEXT_LEN"] = str(
        __supported_context_lengths[
            bisect.bisect_left(__supported_context_lengths, __largest_context)
        ]
    )
    os.environ["VLLM_DT_MAX_BATCH_SIZE"] = str(max(max(COMMON_BATCH_SIZES), 2))
    fx_config.backed_size_oblivious = True

# thresholds are chosen based on 1024 tokens per sequence
# 1% error threshold rate between cpu fp32 and cuda fp16
# if a models failure thresholds do not exist in this dict, default to the default_metrics_threshold defined above
# threshold key is (model_id, is_tiny_model)
FAIL_THRESHOLDS = {
    (LLAMA_3p1_8B_INSTRUCT, False): (
        2.6994638133048965,
        0.00047589250549208347,
    ),
    (GRANITE_3p2_8B_INSTRUCT, False): (
        2.3919514417648315,
        0.0005767398688476533,
    ),
    (GRANITE_3p3_8B_INSTRUCT, False): (
        2.4444521379470827,
        0.0004970188625156878,
    ),
    (GRANITE_20B_CODE_INSTRUCT_8K, False): (
        2.640706129074097,
        0.00034344267623964697,
    ),
    (LLAMA_3p1_70B_INSTRUCT, False): (
        2.841279556751251,
        0.0044301633024588115,
    ),
}

if MODEL_CONFIGURATION_PATH != "":
    print(
        "ignoring FMS_TEST_SHAPES_COMMON_MODEL_PATHS, FMS_TEST_SHAPES_USE_MICRO_MODELS as configuration will be set by FMS_TEST_SHAPES_FROM_MODEL_CONFIGURATION"
    )
    USE_MICRO_MODELS = False
    COMMON_MODEL_PATHS = []
    FREQUENCY = int(MODEL_CONFIGURATION_FREQUENCY)
    with open(MODEL_CONFIGURATION_PATH, "r") as f:
        for line in f:
            try:
                MODEL_CONFIG = json.loads(line)
                if MODEL_CONFIG["frequency"] <= FREQUENCY:
                    COMMON_MODEL_PATHS.append(MODEL_CONFIG["model_id"])
                    # assume fullsize models
                    FAIL_THRESHOLDS[(MODEL_CONFIG["model_id"], USE_MICRO_MODELS)] = (
                        MODEL_CONFIG["ce"],
                        MODEL_CONFIG["mean_diff"],
                    )
            except json.JSONDecodeError:
                print(f"config contained an improper json line: {line.strip()}")

COMMON_SHAPES = list(
    itertools.product(
        COMMON_MODEL_PATHS,
        COMMON_BATCH_SIZES,
        COMMON_SEQ_LENGTHS,
        COMMON_MAX_NEW_TOKENS,
    )
)

# custom weight adaptation to be used in future. For instance if we would like to add some other adaptation, we can register it with this custom adapter
# and provide it when converting from an aiu fms model's weights to a cpu fms model's weights. Currently this is only done for gptq, but may be done for other
# formats in the future
# note: llama already has many adapters for aiu and they are the same for all models, so just use llama. This way we don't need to re-register a new architecture / adapter step (we can just re-use)
__custom_adapter = {"architecture": "llama", "source": "fms_aiu"}


@pytest.fixture(autouse=True)
def reset_compiler():
    yield  # run the test
    if not COMPILE_DYNAMIC_SENDNN:
        torch.compiler.reset()
        torch._dynamo.reset()
        os.environ.pop("COMPILATION_MODE", None)


# TODO: Currently, gptq does not have the same level of support as non-gptq models for get_model. This method provides the extra requirements for gptq for get_model,
#  however ideally, these fixes should be done in foundation-model-stack.
def __maybe_get_gptq_kwargs(model_path):
    gptq_adapter_step = []
    gptq_kwargs_aiu = {}
    gptq_kwargs_cpu = {}
    if GPTQ_ENABLED:
        # TODO: hf_configured/hf_pretrained options in get_model should be inferring the linear_config based on the hf quantization_config attribute
        config = AutoConfig.from_pretrained(model_path)
        if (
            hasattr(config, "quantization_config")
            and config.quantization_config["quant_method"] == "gptq"
        ):
            gptq_adapter_step.append("gptq_qweights_transpose_aiu")
            group_size = config.quantization_config["group_size"]
            desc_act = config.quantization_config["desc_act"]
            linear_config = {"group_size": group_size, "desc_act": desc_act}
            if USE_MICRO_MODELS:
                micro_aiu_kwargs = {"nlayers": 3}
                micro_cpu_kwargs = {"nlayers": 3}
            else:
                # TODO: infer the source based on the device for get_model when using gptq
                micro_aiu_kwargs = {"model_path": model_path, "source": "hf_gptq_aiu"}
                micro_cpu_kwargs = {"model_path": model_path, "source": "hf"}

            # TODO: infer the linear_type based on the device for get_model when using gptq
            gptq_kwargs_aiu = {
                "linear_config": {"linear_type": "gptq_aiu", **linear_config},
                "architecture": "hf_configured",
                "variant": model_path,
                **micro_aiu_kwargs,
            }
            gptq_kwargs_cpu = {
                "linear_config": {"linear_type": "gptq_cpu", **linear_config},
                "architecture": "hf_configured",
                "variant": model_path,
                **micro_cpu_kwargs,
            }
    try:
        # llama already has this adapter and it is the same for all models, so just use llama
        serialization.register_adapter(
            **__custom_adapter, adapter_steps=gptq_adapter_step
        )
    except KeyError:
        pass
    return gptq_kwargs_aiu, gptq_kwargs_cpu


def __prepare_inputs(batch_size, seq_length, tokenizer, model_path, seed=0):
    if "paged" in ATTN_NAME:
        prompts_and_sizes = sample_sharegpt_requests(
            SHARE_GPT_DATASET_PATH,
            batch_size,
            tokenizer,
            32,
            seq_length,
            seed,
            enforce_heterogeneous=True,
            enforce_sizes=[seq_length],  # ensure at least the max seq length is sampled
            pad_multiple=64,
        )
    else:
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
        prompt_list.append(tokenizer.encode(prompt, return_tensors="pt").squeeze(0))

    input_ids, extra_kwargs = pad_input_ids(prompt_list, min_pad_length=seq_length)
    extra_kwargs["attn_name"] = ATTN_NAME
    if (
        "paged" in ATTN_NAME
        and "ibm-granite/granite-3.3-8b-instruct" in model_path
        and USE_DISTRIBUTED
        and dist.get_world_size() == 4
    ):
        extra_kwargs["_kvcache_num_blocks_hint"] = KVCACHE_NUM_BLOCKS_HINT
    return input_ids, extra_kwargs


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


def __filter_before_eos(metrics, filter_indexes):
    from itertools import groupby

    filtered_results = [
        list(g)[: filter_indexes[k]] for k, g in groupby(metrics, key=lambda x: x[0])
    ]
    return [item for sublist in filtered_results for item in sublist]


def __load_validation_info(
    model_path,
    batch_size,
    seq_length,
    max_new_tokens,
    tokenizer,
    seed,
    attn_type: str,
):
    # if path doesn't exist and paged isn't in the attention name, remove `attn_type` and recheck again, warn that we will no longer in the future have paths without 'attn_type'
    full_path = find_validation_info_path(
        VALIDATION_INFO_DIR,
        model_path,
        batch_size,
        seq_length,
        max_new_tokens,
        seed,
        attn_type,
        version_allow_decrement=True,
        dtype=CPU_DTYPE,
    )
    if full_path is not None:
        dprint(f"cpu validation info found for seed={seed} -- loading it")
        return load_validation_information(full_path, "logits", batch_size, tokenizer)
    else:
        return None


class PersistentModel:
    """This class will either get a model that is pre-compiled (if compile_dynamic_sendnn) or re-create the model for each test"""

    def __init__(self):
        self.model = None

    def get_or_create(self, is_gptq, is_fp8, **kwargs):
        if self.model is None:
            model = get_model(
                device_type="cpu",
                data_type=None if is_fp8 or is_gptq else torch.float16,
                fused_weights=False,
                **kwargs,
            )
            self.__maybe_reset_model(model, is_gptq)

            self.__maybe_prepare_fp8_weights(model, is_fp8)

            model.eval()
            model.compile(
                backend="sendnn", options={"sendnn.dynamic": COMPILE_DYNAMIC_SENDNN}
            )

            if COMPILE_DYNAMIC_SENDNN:
                self.model = model

            return model
        else:
            return self.model

    @staticmethod
    def __maybe_prepare_fp8_weights(model, is_fp8):
        if is_fp8:
            for name, param in model.named_parameters():
                if param.dtype == torch.bfloat16:
                    if param.max() > torch.finfo(torch.float16).max:
                        dprint(
                            f"[WARNING] You are casting param {name} to fp16, which will cause loss of accuracy. You can ignore this warning if this is intended."
                        )
                    param.data = param.data.to(dtype=torch.float16)

    # TODO: This was added as we require a special reset for gptq models. Ideally, we would be able to do something like this reset when calling reset_parameters() on the model
    #  however the gptq modules are yet to support this
    @staticmethod
    def __maybe_reset_model(model, is_gptq):
        if USE_MICRO_MODELS and is_gptq:
            sd = model.state_dict()
            for key, param in sd.items():
                if "qweight" in key:
                    res = torch.randint(
                        low=0,
                        high=torch.iinfo(torch.int32).max,
                        size=param.shape,
                        dtype=torch.int32,
                    )
                    sd[key].copy_(res)
                elif "qzeros" in key:
                    res = torch.ones(param.shape, dtype=torch.int32) * 8
                elif "g_idx" in key:
                    res = param
                else:
                    res = torch.randn_like(param)
                    res -= 0.5
                    res /= 20.0
                param.copy_(res)


@pytest.fixture
def persistent_model():
    return PersistentModel()


##### Common utils
# metric calculator based on the cross-entropy and mean diff for each decode step
def _metric_calculator(r: torch.Tensor, t: torch.Tensor):
    cross_entropy = torch.nn.CrossEntropyLoss()(
        r, t.softmax(dim=1).to(dtype=torch.float32)
    )
    diff = torch.mean(
        torch.abs(
            r.softmax(dim=1).to(dtype=torch.float32)
            - t.softmax(dim=1).to(dtype=torch.float32)
        )
    )
    return (cross_entropy, diff)


def _check_failure_thresholds(
    diff_fail_responses_list,
    ce_fail_responses_list,
    total_tokens,
    record_property=None,
):
    # test the failure rates for across all tokens
    diff_failure_rate = len(diff_fail_responses_list) / total_tokens
    ce_failure_rate = len(ce_fail_responses_list) / total_tokens
    dprint(f"mean diff failure rate: {diff_failure_rate}")
    dprint(f"cross entropy loss failure rate: {ce_failure_rate}")

    if record_property is not None:
        # Add failure rates to xml report
        record_property("mean_diff_failure_rate", diff_failure_rate)
        record_property("cross_entropy_loss_failure_rate", ce_failure_rate)

    if "mean_diff" not in SKIP_ASSERTIONS:
        assert diff_failure_rate < FAILURE_RATE_THRESHOLD, (
            f"failure rate for mean diff was too high: {diff_failure_rate}"
        )
    if "ce" not in SKIP_ASSERTIONS:
        assert ce_failure_rate < FAILURE_RATE_THRESHOLD, (
            f"failure rate for cross entropy loss was too high: {ce_failure_rate}"
        )
        print("passed validation level 1")
    else:
        print("passed validation level 0")


def _get_common_model_kwargs(is_gptq, model_path):
    if is_gptq:
        return {}
    # Get the micro model kwargs
    # TODO clean up path handling for micro models
    micro_model_path = MICRO_MODEL_MAPPING.get(model_path, None)

    if USE_MICRO_MODELS and micro_model_path is None:
        dprint("using randomly initialized model")
        micro_model_kwargs = {"architecture": "hf_configured", "nlayers": 3}
    else:
        dprint("using trained model")
        micro_model_kwargs = {"architecture": "hf_pretrained"}

    # Get the model path kwargs
    if not USE_MICRO_MODELS and os.path.exists(model_path):
        model_path_kwargs = {"model_path": model_path}
    elif USE_MICRO_MODELS and micro_model_path is not None:
        model_path_kwargs = {"model_path": micro_model_path}
    else:
        model_path_kwargs = {"variant": model_path}

    # Get the distributed kwargs
    distributed_kwargs = {}
    if USE_DISTRIBUTED:
        distributed_kwargs["distributed_strategy"] = "tp"
        distributed_kwargs["group"] = dist.group.WORLD

    return {
        **model_path_kwargs,
        **micro_model_kwargs,
        **distributed_kwargs,
    }


# NOTE micro_model_state_dict should be None if USE_MICRO_MODELS is true
# Otherwise it should be model.state_dict() where model is the AIU model
def _get_cpu_model(is_gptq, is_fp8, micro_model_state_dict=None, **kwargs):
    # prepare the cpu model
    validation_model = get_model(
        device_type="cpu",
        data_type=None if is_fp8 or is_gptq else torch.float32,
        fused_weights=False,
        **kwargs,
    )

    # This is a micro model, so we need to copy the state dict directly.
    if micro_model_state_dict is not None:
        serialization.load_state_dict_into_model(
            validation_model, micro_model_state_dict, **__custom_adapter
        )
    return validation_model


def _get_device_validation_information(
    model_path,
    batch_size,
    seq_length,
    max_new_tokens,
    post_iteration_hook,
    model,
    input_ids,
    extra_kwargs,
    token_iter,
    device="aiu",
    tokenizer=None,
):
    # For CPU, we try to load it from disk first if it exists
    if device == "cpu":
        cpu_validation_info = __load_validation_info(
            model_path,
            batch_size,
            seq_length,
            max_new_tokens,
            tokenizer,
            token_iter,
            ATTN_NAME,
        )
        if cpu_validation_info is not None:
            return cpu_validation_info

        if cpu_validation_info is not None:
            return cpu_validation_info

    # overrides for validation info that are device specific
    device_dependent_kwargs = {}
    if device == "cpu":
        device_dependent_kwargs["attn_algorithm"] = "math"

    if device == "aiu":
        device_dependent_kwargs["last_n_tokens"] = 64 if "paged" in ATTN_NAME else 1

    # Otherwise we need to get the AIU / CPU validation info
    validation_info = extract_validation_information(
        model,
        input_ids,
        max_new_tokens,
        post_iteration_hook,
        timing=TIMING,
        **extra_kwargs,
        **device_dependent_kwargs,
    )
    if SAVE_VALIDATION_INFO_OUTPUTS:
        dprint(f"saving {device} validation for - iter={token_iter}")
        # TODO - there is probably a cleaner way to handle this too
        kwargs = {}
        if device == "cpu":
            kwargs["dtype"] = CPU_DTYPE

        validation_info.save(
            get_validation_info_path(
                VALIDATION_INFO_DIR,
                model_path,
                batch_size,
                seq_length,
                max_new_tokens,
                token_iter,
                ATTN_NAME,
                device_type=device,
                **kwargs,
            )
        )
    return validation_info


def _resolve_thresholds(model_path, micro_model_path):
    # if we do not have real model weights, use a default_metrics_threshold
    if USE_MICRO_MODELS and micro_model_path is None:
        ce_threshold, diff_threshold = DEFAULT_METRICS_THRESHOLD
    # if we have real weights, try and get the proper validation metrics threshold
    else:
        # if we have a micro model with real weights, but no real thresholds, default to the full model thresholds
        if USE_MICRO_MODELS:
            ce_threshold, diff_threshold = FAIL_THRESHOLDS.get(
                (model_path, True),
                FAIL_THRESHOLDS.get((model_path, False), DEFAULT_METRICS_THRESHOLD),
            )
        else:
            ce_threshold, diff_threshold = FAIL_THRESHOLDS.get(
                (model_path, False), DEFAULT_METRICS_THRESHOLD
            )
    return ce_threshold, diff_threshold


def _run_validation_level_0(
    model_path,
    batch_size,
    seq_length,
    max_new_tokens,
    tokenizer,
    validation_model,
    input_ids,
    extra_kwargs,
    model,
):
    cpu_validation_info = _get_device_validation_information(
        model_path=model_path,
        batch_size=batch_size,
        seq_length=seq_length,
        max_new_tokens=max_new_tokens,
        post_iteration_hook=LogitsExtractorHook(),
        model=validation_model,
        input_ids=input_ids,
        extra_kwargs=extra_kwargs,
        token_iter=0,
        device="cpu",
        tokenizer=tokenizer,
    )

    # Get the cpu static toks / initial eos sequences for iter 0
    cpu_static_tokens = cpu_validation_info.get_info("tokens")
    eos_indexes = __find_eos_index(
        cpu_static_tokens, tokenizer.eos_token_id, seq_length, max_new_tokens
    )
    dprint(
        "cpu validation info extracted for validation level 0 and validation level 1 (iter=0)"
    )

    # first test validation level 0
    aiu_validation_info = _get_device_validation_information(
        model_path=model_path,
        batch_size=batch_size,
        seq_length=seq_length,
        max_new_tokens=max_new_tokens,
        post_iteration_hook=None,
        model=model,
        input_ids=input_ids,
        extra_kwargs=extra_kwargs,
        token_iter=0,
        device="aiu",
        tokenizer=tokenizer,
    )
    dprint("aiu validation info extracted for validation level 0")

    # validate level 0
    failed_responses = validate_level_0(
        aiu_validation_info.get_info("tokens"), cpu_static_tokens
    )

    # Keep things we may need on the first iter for validation 1
    validation_zero_info = {
        "cpu_validation_info": cpu_validation_info,
        "cpu_static_tokens": cpu_static_tokens,
        "eos_indexes": eos_indexes,
    }
    return len(failed_responses) != 0, validation_zero_info


def _run_validation_level_1(
    model_path,
    batch_size,
    seq_length,
    max_new_tokens,
    tokenizer,
    validation_model,
    input_ids,
    extra_kwargs,
    model,
    micro_model_path,
    validation_zero_info,
    record_property,
):
    iters = int(CUMULATIVE_TEST_TOKENS_PER_SEQUENCE) // max_new_tokens
    ce_fail_responses_list = []
    diff_fail_responses_list = []
    total_tokens = 0
    for i in range(iters):
        # for iteration 0, we have computed the cpu validation info in the prior step for seed=0, so skip
        if i != 0:
            input_ids, extra_kwargs = __prepare_inputs(
                batch_size, seq_length, tokenizer, model_path, seed=i
            )
            cpu_validation_info = _get_device_validation_information(
                model_path=model_path,
                batch_size=batch_size,
                seq_length=seq_length,
                max_new_tokens=max_new_tokens,
                post_iteration_hook=LogitsExtractorHook(),
                model=validation_model,
                input_ids=input_ids,
                extra_kwargs=extra_kwargs,
                token_iter=i,
                device="cpu",
                tokenizer=tokenizer,
            )
            dprint(f"cpu validation info extracted for validation level 1 - iter={i}")

            cpu_static_tokens = cpu_validation_info.get_info("tokens")
            eos_indexes = __find_eos_index(
                cpu_static_tokens,
                tokenizer.eos_token_id,
                seq_length,
                max_new_tokens,
            )
        else:
            # TODO this can be cleaned up further
            cpu_validation_info = validation_zero_info["cpu_validation_info"]
            cpu_static_tokens = validation_zero_info["cpu_static_tokens"]
            eos_indexes = validation_zero_info["eos_indexes"]

        aiu_validation_info = _get_device_validation_information(
            model_path=model_path,
            batch_size=batch_size,
            seq_length=seq_length,
            max_new_tokens=max_new_tokens,
            post_iteration_hook=GoldenTokenHook(cpu_static_tokens),
            model=model,
            input_ids=input_ids,
            extra_kwargs=extra_kwargs,
            token_iter=i,
            device="aiu",
            tokenizer=tokenizer,
        )
        dprint(f"aiu validation info extracted for validation level 1 - iter={i}")

        # capture all level 1 metrics
        level_1_metrics = capture_level_1_metrics(
            cpu_validation_info.get_info("logits"),
            aiu_validation_info.get_info("logits"),
            top_k_loss_calculator(20, _metric_calculator),
        )
        # only consider those metrics captured prior to the eos
        level_1_metrics = __filter_before_eos(level_1_metrics, eos_indexes)

        ce_threshold, diff_threshold = _resolve_thresholds(model_path, micro_model_path)

        # get all failed responses for each metric
        ce_fail_responses = filter_failed_level_1_cases(
            level_1_metrics, lambda m: m[0] >= ce_threshold
        )
        diff_fail_responses = filter_failed_level_1_cases(
            level_1_metrics,
            lambda m: m[1] >= diff_threshold,
        )

        ce_fail_responses_list.extend(ce_fail_responses)
        diff_fail_responses_list.extend(diff_fail_responses)
        total_tokens += len(level_1_metrics)

    _check_failure_thresholds(
        diff_fail_responses_list,
        ce_fail_responses_list,
        total_tokens,
        record_property,
    )


##### Test definitions
def _run_cpu_aiu_validation_test(
    model_path,
    batch_size,
    seq_length,
    max_new_tokens,
    cpu_model,
    aiu_model,
    micro_model_path,
    record_property,
    verify_cache_state=None,
):
    # Get the tokenizer and AIU / CPU models to compare
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # prepare input_ids
    input_ids, extra_kwargs = __prepare_inputs(
        batch_size, seq_length, tokenizer, model_path
    )

    extra_kwargs["attn_name"] = ATTN_NAME
    if (
        "paged" in ATTN_NAME
        and "ibm-granite/granite-3.3-8b-instruct" in model_path
        and USE_DISTRIBUTED
        and dist.get_world_size() == 4
    ):
        extra_kwargs["_kvcache_num_blocks_hint"] = KVCACHE_NUM_BLOCKS_HINT

    # warmup aiu model
    warmup_model(
        aiu_model, input_ids, max_new_tokens, COMPILE_DYNAMIC_SENDNN, **extra_kwargs
    )

    # Run validation level 0
    failed_validation_level_0, validation_zero_info = _run_validation_level_0(
        model_path,
        batch_size,
        seq_length,
        max_new_tokens,
        tokenizer,
        cpu_model,
        input_ids,
        extra_kwargs,
        aiu_model,
    )

    # Used only for cache tests; this is a nonparametric closure that
    # should assert the cache for torch sendnn is in the correct state
    # for this test
    if verify_cache_state is not None:
        verify_cache_state()

    # if level 0 fails validation, validate level 1
    if FORCE_VALIDATION_LEVEL_1 or failed_validation_level_0:
        if failed_validation_level_0:
            dprint("failed validation level 0, testing validation level 1")
        else:
            dprint("passed validation level 0, testing validation level 1")
        _run_validation_level_1(
            model_path,
            batch_size,
            seq_length,
            max_new_tokens,
            tokenizer,
            cpu_model,
            input_ids,
            extra_kwargs,
            aiu_model,
            micro_model_path,
            validation_zero_info,
            record_property,
        )


def _get_cache_test_params():
    # NOTE - currently we always use granite 3.3 for the cache test,
    # TODO make this configurable as tests are refactored
    model_path = GRANITE_3p3_8B_INSTRUCT
    batch_size = COMMON_BATCH_SIZES[0]
    seq_length = COMMON_SEQ_LENGTHS[0]
    max_new_tokens = COMMON_MAX_NEW_TOKENS[0]
    return [model_path, batch_size, seq_length, max_new_tokens]


def _reset_cache_settings(purge_cache_dir):
    os.environ["TORCH_SENDNN_CACHE_ENABLE"] = "1"
    os.environ["COMPILATION_MODE"] = "offline_decoder"
    cache_dir = os.environ["TORCH_SENDNN_CACHE_DIR"]

    # Ensure we start in clean state
    if purge_cache_dir and os.path.isdir(cache_dir):
        shutil.rmtree(cache_dir)
        os.mkdir(cache_dir)

        _get_global_state().use_aiu_cache = True
        _get_global_state().spyre_graph_cache = SpyreGraphCache()


@pytest.fixture
def use_cached_model(persistent_model, record_property):
    """Configures the torchsendnn cache and runs the AIU model prior to test execution;
    this is computationally expensive and should only be used in situations like testing
    cache hit correctness;
    """
    torch.manual_seed(42)
    torch.set_grad_enabled(False)
    _reset_cache_settings(purge_cache_dir=True)

    model_path, batch_size, seq_length, max_new_tokens = _get_cache_test_params()
    micro_model_path = MICRO_MODEL_MAPPING.get(model_path, None)

    def verify_cache_miss():
        cache_dir = os.environ.get("TORCH_SENDNN_CACHE_DIR")
        updated_cache_len = (
            len(os.listdir(cache_dir)) if os.path.isdir(cache_dir) else 0
        )
        assert updated_cache_len == max_new_tokens, (
            "cache directory not populated on cache miss"
        )

    dprint(
        f"Setting up cache [i.e., cache miss check] for model={model_path}, batch_size={batch_size}, seq_length={seq_length}, max_new_tokens={max_new_tokens}, micro_model={USE_MICRO_MODELS}"
    )

    # we don't currently support inferring gptq from get_model, so we must use an adapter with hf_configured
    gptq_kwargs_aiu, gptq_kwargs_cpu = __maybe_get_gptq_kwargs(model_path)
    is_gptq = len(gptq_kwargs_aiu) != 0
    is_fp8 = "fp8" in ATTN_NAME
    model_kwargs = _get_common_model_kwargs(is_gptq, model_path)

    # Get the AIU model w/ the persistent model fixture
    model = persistent_model.get_or_create(
        is_gptq, is_fp8, **gptq_kwargs_aiu, **model_kwargs
    )

    validation_model = _get_cpu_model(
        is_gptq,
        is_fp8,
        micro_model_state_dict=model.state_dict() if USE_MICRO_MODELS else None,
        **gptq_kwargs_cpu,
        **model_kwargs,
    )

    _run_cpu_aiu_validation_test(
        model_path,
        batch_size,
        seq_length,
        max_new_tokens,
        validation_model,
        model,
        micro_model_path,
        record_property,
        verify_cache_state=verify_cache_miss,
    )


@pytest.mark.parametrize(
    "model_path,batch_size,seq_length,max_new_tokens", COMMON_SHAPES
)
def test_common_shapes(
    model_path,
    batch_size,
    seq_length,
    max_new_tokens,
    persistent_model,
    record_property,
):
    torch.manual_seed(42)
    torch.set_grad_enabled(False)
    os.environ["COMPILATION_MODE"] = "offline_decoder"
    micro_model_path = MICRO_MODEL_MAPPING.get(model_path, None)

    dprint(
        f"testing model={model_path}, batch_size={batch_size}, seq_length={seq_length}, max_new_tokens={max_new_tokens}, micro_model={USE_MICRO_MODELS}, attn_type={ATTN_TYPE}"
    )

    # we don't currently support inferring gptq from get_model, so we must use an adapter with hf_configured
    gptq_kwargs_aiu, gptq_kwargs_cpu = __maybe_get_gptq_kwargs(model_path)
    is_gptq = len(gptq_kwargs_aiu) != 0
    is_fp8 = "fp8" in ATTN_NAME
    model_kwargs = _get_common_model_kwargs(is_gptq, model_path)

    # Get the AIU model w/ the persistent model fixture
    model = persistent_model.get_or_create(
        is_gptq, is_fp8, **gptq_kwargs_aiu, **model_kwargs
    )

    validation_model = _get_cpu_model(
        is_gptq,
        is_fp8,
        micro_model_state_dict=model.state_dict() if USE_MICRO_MODELS else None,
        **gptq_kwargs_cpu,
        **model_kwargs,
    )

    _run_cpu_aiu_validation_test(
        model_path,
        batch_size,
        seq_length,
        max_new_tokens,
        validation_model,
        model,
        micro_model_path,
        record_property,
    )


def test_cache(use_cached_model, persistent_model, record_property):
    torch.manual_seed(42)
    torch.set_grad_enabled(False)
    _reset_cache_settings(purge_cache_dir=False)

    model_path, batch_size, seq_length, max_new_tokens = _get_cache_test_params()
    micro_model_path = MICRO_MODEL_MAPPING.get(model_path, None)

    def verify_cache_hit():
        cache_dir = os.environ.get("TORCH_SENDNN_CACHE_DIR")
        updated_cache_len = (
            len(os.listdir(cache_dir)) if os.path.isdir(cache_dir) else 0
        )
        assert updated_cache_len == max_new_tokens, (
            "cache miss occurred when hit was expected"
        )

    dprint(
        f"testing: model={model_path}, batch_size={batch_size}, seq_length={seq_length}, max_new_tokens={max_new_tokens}, micro_model={USE_MICRO_MODELS}, for cache hit"
    )

    # we don't currently support inferring gptq from get_model, so we must use an adapter with hf_configured
    gptq_kwargs_aiu, gptq_kwargs_cpu = __maybe_get_gptq_kwargs(model_path)
    is_gptq = len(gptq_kwargs_aiu) != 0
    is_fp8 = "fp8" in ATTN_NAME
    model_kwargs = _get_common_model_kwargs(is_gptq, model_path)

    # Get the AIU model w/ the persistent model fixture
    model = persistent_model.get_or_create(
        is_gptq, is_fp8, **gptq_kwargs_aiu, **model_kwargs
    )

    validation_model = _get_cpu_model(
        is_gptq,
        is_fp8,
        micro_model_state_dict=model.state_dict() if USE_MICRO_MODELS else None,
        **gptq_kwargs_cpu,
        **model_kwargs,
    )

    _run_cpu_aiu_validation_test(
        model_path,
        batch_size,
        seq_length,
        max_new_tokens,
        validation_model,
        model,
        micro_model_path,
        record_property,
        verify_cache_state=verify_cache_hit,
    )
