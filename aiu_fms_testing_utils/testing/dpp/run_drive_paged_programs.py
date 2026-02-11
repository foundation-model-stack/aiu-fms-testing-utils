import datetime
import os
from torch.fx.experimental import _config as fx_config
from aiu_fms_testing_utils.testing.dpp.generation import (
    generate_aiu_cpu_test,
    generate_aiu_test,
)
from aiu_fms_testing_utils.testing.dpp.prepare_data import (
    prepare_test_prompts,
    resolve_dataset_path,
)
from aiu_fms_testing_utils.testing.dpp.prepare_model import (
    _get_model_kwargs,
    load_model,
)
from aiu_fms_testing_utils.testing.dpp.program_models import EnvConfig
from aiu_fms_testing_utils.testing.dpp.sample_prompts import get_sampler
from aiu_fms_testing_utils.utils import warmup_model
from aiu_fms_testing_utils.utils.aiu_setup import (
    aiu_dist_setup,
    dprint,
    r0dprint,
    local_rank,
)
from aiu_fms_testing_utils.utils.dpp_config import DPPRunnerConfig
from aiu_fms_testing_utils.testing.dpp.constants import PAD_MULTIPLE
from aiu_fms_testing_utils.utils.model_setup import Timing
from aiu_fms_testing_utils.testing.dpp.program_models import (
    DeviceType,
    TestType,
    AttnType,
)

from fms.utils.generation import pad_input_ids

import torch
from torch import distributed as dist
from transformers import AutoTokenizer


from typing import Any, Dict, List


DEFAULT_CE_THRESHOLD = 2.5
DEFAULT_FAILURE_RATE_THRESHOLD = 0.1
SHARE_GPT_DATASET = (
    "anon8231489123/ShareGPT_Vicuna_unfiltered",
    "ShareGPT_V3_unfiltered_cleaned_split.json",
)
RAG_FACTOID_DATASET = ("", "")


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


def setup_environment(
    program_criteria_json_path: str, attention_type: AttnType
) -> EnvConfig:
    """Set up global process state and environment variables.

    Args:
        program_criteria_json_path: Path to the JSON file containing program criteria definitions.
        attention_type: Type of attention mechanism to use.

    Returns:
        EnvConfig: Immutable configuration containing:
            - attn_type: Mapped attention implementation name
            - cpu_dtype: Data type for CPU operations ("fp8" or "fp32")
            - max_batch_size: Maximum batch size from VLLM_DT_MAX_BATCH_SIZE
            - max_tkv: Maximum token-key-value context length from VLLM_DT_MAX_CONTEXT_LEN

    Raises:
        ValueError: If required environment variables VLLM_DT_MAX_CONTEXT_LEN or
                    VLLM_DT_MAX_BATCH_SIZE are not set."""

    os.environ["COMPILATION_MODE"] = "offline_decoder"
    os.environ["DT_PROG_CRITERIA_FILEPATH"] = program_criteria_json_path

    if (
        "VLLM_DT_MAX_CONTEXT_LEN" not in os.environ
        or "VLLM_DT_MAX_BATCH_SIZE" not in os.environ
    ):
        r0dprint("Missing required VLLM environment variables.")
        raise ValueError(
            "Environment variables VLLM_DT_MAX_CONTEXT_LEN and VLLM_DT_MAX_BATCH_SIZE must be set before running DPP."
        )

    torch.manual_seed(42)
    torch.set_grad_enabled(False)
    fx_config.backed_size_oblivious = True

    return EnvConfig(
        attn_type=attention_type,
        cpu_dtype="fp8" if "fp8" in attention_type.value else "fp32",
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
    timing: Timing = Timing.NONE,
    test_type: TestType = TestType.METRICS,
    cross_entropy_threshold: float = DEFAULT_CE_THRESHOLD,
    failure_rate_threshold: float = DEFAULT_FAILURE_RATE_THRESHOLD,
    attention_type: AttnType = AttnType.PAGED,
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

    is_fp8 = attention_type == AttnType.PAGED_FP8
    if not run_cpu_validation and test_type == TestType.METRICS:
        dprint("When skipping validation, only test_type will be ignored")

    # Environment Setup
    env_config = setup_environment(
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
    model_kwargs = _get_model_kwargs(model_variant)
    distributed_kwargs = _get_distributed_kwargs(dist_timeout) if distributed else {}

    # Setup model config
    model_config = DPPRunnerConfig()
    model_config.setup_config(
        model_variant=model_variant,
        use_distributed=distributed,
        prefill_chunk_size=prefill_chunk_size,
    )

    model = load_model(
        device_type=DeviceType.SPYRE,
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
    extra_kwargs["attn_name"] = env_config.attn_type.value
    extra_kwargs["_kvcache_num_blocks_hint"] = model_config.num_blocks
    warmup_model(
        model,
        input_ids,
        max_new_tokens,
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
        program_criteria_json_path,
        programs,
        max_new_tokens,
        prioritize_large_batch_sizes,
        enforce_homogeneous_prompt_programs,
        env_config.max_batch_size,
        env_config.max_tkv,
        model_config.tkv_limit,
        tokenizer,
        sampler,
        allow_truncation,
        custom_shape,
        local_dataset_path,
    )

    # Validation and Testing
    if run_cpu_validation:
        validation_model = load_model(
            device_type=DeviceType.CPU,
            is_fp8=is_fp8,
            model_kwargs=model_kwargs,
            distributed_kwargs=distributed_kwargs,
            stagger_load=stagger_load,
            model_config=model_config,
        )
        save_validation_info_outputs = (
            save_validation_info_outputs and dist.get_rank() == 0
        )

        failed_cases = generate_aiu_cpu_test(
            valid_prompts,
            model,
            validation_model,
            tokenizer,
            env_config,
            model_config,
            test_type,
            max_new_tokens,
            save_validation_info_outputs,
            validation_info_outputs_dir,
            cross_entropy_threshold,
            failure_rate_threshold,
            timing,
            prefill_chunk_size,
            model_variant,
        )

        if local_rank != 0:
            return

        if len(failed_cases) != 0:
            dprint("The test failed with the following cases:")
            for failed_case in failed_cases:
                dprint(
                    f"Program ID: {failed_case[0]}, Prompt Shape: {failed_case[1]}, Failure Rate: {failed_case[2]}"
                )
        else:
            dprint("All tests passed")
    else:
        generate_aiu_test(
            valid_prompts,
            model,
            tokenizer,
            env_config,
            model_config,
            test_type,
            max_new_tokens,
            timing,
            prefill_chunk_size,
        )
