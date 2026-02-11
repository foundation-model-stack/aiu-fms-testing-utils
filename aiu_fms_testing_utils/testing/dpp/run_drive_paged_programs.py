import datetime
import os
from huggingface_hub import hf_hub_download
from torch.fx.experimental import _config as fx_config
from aiu_fms_testing_utils.testing.dpp.generation import (
    generate_validation_info_and_test,
)
from aiu_fms_testing_utils.testing.dpp.prepare_prompts import prepare_test_prompts
from aiu_fms_testing_utils.testing.dpp.program_models import EnvConfig
from aiu_fms_testing_utils.testing.dpp.sample_prompts import get_sampler
from aiu_fms_testing_utils.utils import stagger_region, warmup_model
from aiu_fms_testing_utils.utils.aiu_setup import aiu_dist_setup, dprint, local_rank
from aiu_fms_testing_utils.utils.dpp_config import DPPRunnerConfig
from aiu_fms_testing_utils.utils.env_utils import scoped_environ
from aiu_fms_testing_utils.testing.dpp.constants import PAD_MULTIPLE

from fms.models import get_model
from fms.utils.generation import pad_input_ids

import torch
from torch import distributed as dist
from transformers import AutoTokenizer


from typing import Any, Dict, List, Literal


DEFAULT_CE_THRESHOLD = 2.5
DEFAULT_FAILURE_RATE_THRESHOLD = 0.1
SHARE_GPT_DATASET = (
    "anon8231489123/ShareGPT_Vicuna_unfiltered",
    "ShareGPT_V3_unfiltered_cleaned_split.json",
)
RAG_FACTOID_DATASET = ("", "")


def resolve_dataset_path(dataset_path: str) -> tuple[str, str]:
    """Resolves the dataset type and local path based on the provided dataset_path.

    Args:
        dataset_path: A string indicating the dataset to use. Supported values are:
                      - "sharegpt": Uses the ShareGPT dataset from HuggingFace.
                      - "rag_factoid": Uses the RAG Factoid dataset from HuggingFace.
                      - Any other string is considered a custom dataset path.

    Returns:
        A tuple containing:
            - dataset_type: A string indicating the type of dataset ("sharegpt", "rag_factoid", or "custom").
            - local_dataset_path: The local file path to the dataset."""

    if dataset_path == "sharegpt":
        dataset_type = "sharegpt"
        # Fetch from HuggingFace
        local_dataset_path = hf_hub_download(
            repo_id=SHARE_GPT_DATASET[0],
            filename=SHARE_GPT_DATASET[1],
            repo_type="dataset",
        )
    elif dataset_path == "rag_factoid":
        dataset_type = "rag_factoid"
        local_dataset_path = hf_hub_download(
            repo_id=RAG_FACTOID_DATASET[0],
            filename=RAG_FACTOID_DATASET[1],
            repo_type="dataset",
        )
    elif dataset_path is None:
        dataset_type = "custom"
        local_dataset_path = dataset_path
    else:
        raise ValueError(f"Unsupported dataset_path: {dataset_path}")

    return dataset_type, local_dataset_path


def _get_distributed_kwargs(dist_timeout: str) -> Dict[str, Any]:
    """Initializes distributed training configuration and returns kwargs.

    Sets up PyTorch distributed process group with tensor parallelism strategy. Configures custom timeout if specified.

    Args:
        dist_timeout: Timeout in minutes for distributed operations (0 uses default).

    Returns:
        Dictionary containing distributed configuration with keys:
            - "distributed_strategy": Set to "tp" (tensor parallelism) if distributed.
            - "group": PyTorch distributed group (WORLD) if distributed."""

    if dist_timeout > 0:
        # Default timeout:
        # https://docs.pytorch.org/docs/stable/distributed.html#torch.distributed.init_process_group
        dist.init_process_group(timeout=datetime.timedelta(minutes=dist_timeout))
        dprint(f"NOTICE: init_process_group timeout set to {dist_timeout} minutes")
    else:
        dist.init_process_group()

    aiu_dist_setup(dist.get_rank(), dist.get_world_size())
    return {
        "distributed_strategy": "tp",
        "group": dist.group.WORLD,
    }


def _get_model_kwargs(model_variant: str) -> Dict[str, Any]:
    """Constructs model loading kwargs based on whether variant is a path or ID.

    Determines if the model_variant is a local filesystem path or a HuggingFace
    model identifier, and returns the appropriate keyword arguments for model loading.

    Args:
        model_variant: Either a local path to model files or a HuggingFace model ID.

    Returns:
        Dictionary with either "model_path" (for local paths) or "variant"
        (for HuggingFace IDs) as the key."""

    model_kwargs = {}
    if os.path.exists(model_variant):
        model_kwargs["model_path"] = model_variant
    else:
        model_kwargs["variant"] = model_variant

    return model_kwargs


def _maybe_prepare_fp8_weights(model: torch.nn.Module) -> None:
    """Converts model weights from bfloat16 to float16 for FP8 attention.

    When using FP8 attention variants, this function converts all bfloat16 parameters
    to float16. Issues a warning if any parameter values exceed the float16 range,
    which may cause accuracy loss.

    Args:
        model: PyTorch model whose weights may need conversion."""

    for name, param in model.named_parameters():
        if param.dtype == torch.bfloat16:
            if param.max() > torch.finfo(torch.float16).max:
                dprint(
                    f"[WARNING] You are casting param {name} to fp16, which will cause loss of accuracy."
                )
            param.data = param.data.to(dtype=torch.float16)


def load_model(
    device_type: Literal["cpu", "spyre"],
    is_fp8: bool,
    model_kwargs: Dict[str, Any],
    distributed_kwargs: Dict[str, Any],
    stagger_load: int,
    model_config: DPPRunnerConfig,
):
    """Loads and optionally compiles a model for inference or validation.

    Loads a model with the specified configuration. For Spyre/AIU models,
    compiles the model using the sendnn backend with dynamic compilation enabled.
    The scoped_environ context manager temporarily sets environment variables
    from model_config during compilation to configure the compiler's behavior (e.g.,
    program criteria, batch sizes, context lengths).

    Args:
        device_type: Target device for model execution. Options:
            - "cpu": Load on CPU for validation (fp32, no compilation)
            - "spyre": Load on CPU, compile for Spyre/AIU execution (fp16, with sendnn compilation)
        is_fp8: If True, uses FP8 quantization (dtype=None for auto-detection).
        model_kwargs: Dictionary with model loading parameters (variant or path).
        distributed_kwargs: Dictionary with distributed training configuration.
        stagger_load: Number of concurrent processes allowed during loading (0=unlimited).
        model_config: DPPRunnerConfig instance with environment variable updates.

    Returns:
        torch.nn.Module: Loaded model in evaluation mode. Spyre models are compiled
        with sendnn backend and may have FP8 weight conversion applied.
    """

    if device_type not in ["cpu", "spyre"]:
        raise ValueError(
            f"device_type must be 'cpu' or 'spyre' for DPP, got '{device_type}'"
        )

    dtype = (
        (torch.float32 if device_type == "cpu" else torch.float16)
        if not is_fp8
        else None
    )

    with stagger_region(stagger_load):
        model = get_model(
            architecture="hf_pretrained",
            device_type="cpu",
            data_type=dtype,
            fused_weights=False,
            **model_kwargs,
            **distributed_kwargs,
        )

    model.eval()

    if device_type == "spyre":
        with scoped_environ(model_config.env_updates()):
            # Temporarily set environment variables needed for compile
            model.compile(backend="sendnn", options={"sendnn.dynamic": True})

        if is_fp8:
            _maybe_prepare_fp8_weights(model)

    return model


def setup_environment(
    program_criteria_json_path: str, attention_type: str
) -> EnvConfig:
    """Set up global process state and environment variables.

    Args:
        program_criteria_json_path: Path to the JSON file containing program criteria definitions.
        attention_type: Type of attention mechanism to use. Must be one of sdpa, paged, math_fp8, paged_fp8.

    Returns:
        EnvConfig: Immutable configuration containing:
            - attn_name: Mapped attention implementation name
            - cpu_dtype: Data type for CPU operations ("fp8" or "fp32")
            - max_batch_size: Maximum batch size from VLLM_DT_MAX_BATCH_SIZE
            - max_tkv: Maximum token-key-value context length from VLLM_DT_MAX_CONTEXT_LEN

    Raises:
        SystemExit: If required environment variables VLLM_DT_MAX_CONTEXT_LEN or
                    VLLM_DT_MAX_BATCH_SIZE are not set.
    """
    os.environ["COMPILATION_MODE"] = "offline_decoder"
    os.environ["DT_PROG_CRITERIA_FILEPATH"] = program_criteria_json_path

    if (
        "VLLM_DT_MAX_CONTEXT_LEN" not in os.environ
        or "VLLM_DT_MAX_BATCH_SIZE" not in os.environ
    ):
        if local_rank == 0:
            dprint("Missing required VLLM environment variables.")
        exit(1)

    torch.manual_seed(42)
    torch.set_grad_enabled(False)
    fx_config.backed_size_oblivious = True

    attention_map = {
        "sdpa": "sdpa_causal",
        "paged": "spyre_paged_attn",
        "math_fp8": "math_fp8",
        "paged_fp8": "spyre_paged_attn_fp8",
    }

    return EnvConfig(
        attn_name=attention_map[attention_type],
        cpu_dtype="fp8" if "fp8" in attention_type else "fp32",
        max_batch_size=int(os.environ["VLLM_DT_MAX_BATCH_SIZE"]),
        max_tkv=int(os.environ["VLLM_DT_MAX_CONTEXT_LEN"]),
    )


def run_dpp(
    program_criteria_json_path: str,
    dataset_path: str,
    max_new_tokens: int,
    distributed: bool,
    model_variant: str,
    programs: List[str] = None,
    timing: str = "",
    test_type: str = "metrics",
    cross_entropy_threshold: float = DEFAULT_CE_THRESHOLD,
    failure_rate_threshold: float = DEFAULT_FAILURE_RATE_THRESHOLD,
    attention_type: str = "paged",
    prefill_chunk_size: int = 0,
    stagger_load: int = 0,
    stagger_update_lazyhandle: int = 0,
    dist_timeout: int = 0,
    run_cpu_validation: bool = True,
    validation_info_outputs_dir: str = "/home/senuser/models/validation_info",
    save_validation_info_outputs: bool = False,
    prioritize_large_batch_sizes: bool = False,
    enforce_homogeneous_prompt_programs: bool = False,
):

    if programs is None:
        programs = []

    dataset_type, local_dataset_path = resolve_dataset_path(dataset_path)

    # Environment Setup
    is_fp8: bool = attention_type == "paged_fp8"
    if not run_cpu_validation and test_type == "metrics":
        dprint("When skipping validation, only test_type will be ignored")
    env_config: EnvConfig = setup_environment(
        program_criteria_json_path=program_criteria_json_path,
        attention_type=attention_type,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_variant)
    sampler, allow_truncation, custom_shape = get_sampler(
        dataset_type=dataset_type,
        dataset_path=local_dataset_path,
        tokenizer=tokenizer,
    )

    # Model Loading
    model_kwargs: Dict[str, Any] = _get_model_kwargs(model_variant=model_variant)
    distributed_kwargs: Dict[str, Any] = (
        _get_distributed_kwargs(dist_timeout) if distributed else {}
    )
    save_validation_info_outputs = save_validation_info_outputs and dist.get_rank() == 0
    model_config: DPPRunnerConfig = DPPRunnerConfig()
    world_size = dist.get_world_size() if distributed and dist.is_initialized() else 1
    model_config.setup_config(
        model_variant=model_variant,
        use_distributed=distributed,
        world_size=world_size,
        prefill_chunk_size=prefill_chunk_size,
    )

    model = load_model(
        device_type="spyre",
        is_fp8=is_fp8,
        model_kwargs=model_kwargs,
        distributed_kwargs=distributed_kwargs,
        stagger_load=stagger_load,
        model_config=model_config,
    )

    validation_model = None
    if run_cpu_validation:
        validation_model = load_model(
            device_type="cpu",
            is_fp8=is_fp8,
            model_kwargs=model_kwargs,
            distributed_kwargs=distributed_kwargs,
            stagger_load=stagger_load,
            model_config=model_config,
        )

    # Model Warmup
    ## warmup with any input so compiler produces criteria json
    ## TODO: Swap this with _prepare_inputs once fix for shape_id is available
    ## input_ids, extra_kwargs, sample_key = _prepare_inputs(2, max_tkv, tokenizer)
    prompt_list = [torch.arange(0, PAD_MULTIPLE, dtype=torch.int64)]
    # matching vllm warmup to pad to 2 on fp8, and no pad for fp16
    if is_fp8:
        prompt_list = prompt_list * 2
    input_ids, extra_kwargs = pad_input_ids(prompt_list, min_pad_length=64)
    extra_kwargs["mask"] = extra_kwargs["mask"].to(torch.float16)
    extra_kwargs["attn_name"] = env_config.attn_name
    extra_kwargs["_kvcache_num_blocks_hint"] = model_config.num_blocks
    warmup_model(
        model=model,
        input_ids=input_ids,
        max_new_tokens=max_new_tokens,
        compile_dynamic_sendnn=True,
        stagger_update_lazyhandle=stagger_update_lazyhandle,
        prefill_chunk_size=prefill_chunk_size,
        **extra_kwargs,
    )
    if distributed:
        # wait for rank0 to be finished as it is the only one generating the criteria json
        # this is needed since otherwise we may run into a race condition
        torch.distributed.barrier()

    # Prompt Preparation
    valid_prompts = prepare_test_prompts(
        program_criteria_json_path=program_criteria_json_path,
        programs=programs,
        max_new_tokens=max_new_tokens,
        prioritize_large_batch_sizes=prioritize_large_batch_sizes,
        enforce_homogeneous_prompt_programs=enforce_homogeneous_prompt_programs,
        max_batch_size=env_config.max_batch_size,
        max_tkv=env_config.max_tkv,
        tkv_limit=model_config.tkv_limit,
        tokenizer=tokenizer,
        sampler=sampler,
        allow_truncation=allow_truncation,
        custom_shape=custom_shape,
        dataset_path=local_dataset_path,
    )

    # Validation and Testing
    failed_cases = generate_validation_info_and_test(
        valid_prompts=valid_prompts,
        model=model,
        validation_model=validation_model,
        tokenizer=tokenizer,
        env_config=env_config,
        model_config=model_config,
        test_type=test_type,
        max_new_tokens=max_new_tokens,
        skip_validation=not run_cpu_validation,
        save_validation_info_outputs=save_validation_info_outputs,
        validation_info_outputs_dir=validation_info_outputs_dir,
        cross_entropy_threshold=cross_entropy_threshold,
        failure_rate_threshold=failure_rate_threshold,
        timing=timing,
        prefill_chunk_size=prefill_chunk_size,
        model_variant=model_variant,
    )

    if run_cpu_validation and local_rank == 0:
        if len(failed_cases) != 0:
            dprint("The test failed with the following cases:")
            for failed_case in failed_cases:
                dprint(
                    f"Program ID: {failed_case[0]}, Prompt Shape: {failed_case[1]}, Failure Rate: {failed_case[2]}"
                )
        else:
            dprint("all tests passed")
