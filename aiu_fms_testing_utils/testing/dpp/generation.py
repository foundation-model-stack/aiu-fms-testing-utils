from aiu_fms_testing_utils.testing.dpp.metrics_validation import (
    evaluate_cross_entropy_metrics,
    report_token_comparison,
)
from aiu_fms_testing_utils.testing.dpp.program_models import EnvConfig, ValidPrompt
from aiu_fms_testing_utils.testing.validation import (
    GoldenTokenHook,
    LogitsExtractorHook,
    ValidationInfo,
    extract_validation_information,
    get_validation_info_path,
)
from aiu_fms_testing_utils.utils.aiu_setup import dprint, local_rank
from aiu_fms_testing_utils.utils.dpp_config import DPPRunnerConfig
from aiu_fms_testing_utils.testing.dpp.metrics_validation import _load_validation_info
from aiu_fms_testing_utils.utils.model_setup import Timing
from aiu_fms_testing_utils.testing.dpp.program_models import TestType, AttnType


import torch
from transformers import AutoTokenizer


from typing import Any, Iterable, Optional


def generate_aiu_validation(
    test_type: TestType,
    max_new_tokens: int,
    timing: Timing,
    prefill_chunk_size: int,
    model: torch.nn.Module,
    valid_prompt: ValidPrompt,
    cpu_validation_info: Optional[ValidationInfo] = None,
) -> ValidationInfo:
    """Generates AIU validation information by running inference on the compiled model.

    Executes the AIU-compiled model to generate tokens and extract logits. If CPU
    validation info is available and test_type is "metrics", injects golden tokens
    from CPU validation to ensure consistent decode paths for metric comparison.

    Args:
        test_type: Type of test being run.
        max_new_tokens: Maximum number of tokens to generate.
        timing: Whether to collect timing information.
        prefill_chunk_size: Chunk size for prefill operations.
        model: Compiled AIU model for inference.
        valid_prompt: ValidPrompt object containing prompt input IDs and extra kwargs for model execution.
        cpu_validation_info: Optional CPU validation data for golden token injection.

    Returns:
        ValidationInfo: ValidationInfo object containing AIU outputs (tokens, logits,
        and optional timing information)."""

    golden_hook = None
    if test_type == TestType.METRICS and cpu_validation_info is not None:
        golden_hook = GoldenTokenHook(cpu_validation_info.get_info("tokens"))

    aiu_validation_info = extract_validation_information(
        model=model,
        input_ids=valid_prompt.input_ids,
        max_new_tokens=max_new_tokens,
        post_iteration_hook=golden_hook,
        last_n_tokens=64,
        timing=timing,
        prefill_chunk_size=prefill_chunk_size,
        **valid_prompt.extra_kwargs,
    )

    return aiu_validation_info


def generate_cpu_validation(
    model_variant: str,
    max_new_tokens: int,
    validation_info_outputs_dir: str,
    save_validation_info_outputs: bool,
    validation_model: torch.nn.Module,
    valid_prompt: ValidPrompt,
    env_config: EnvConfig,
    tokenizer: AutoTokenizer,
) -> ValidationInfo:
    """Generates or loads CPU validation information for reference comparison.

    Attempts to load pre-computed CPU validation data from disk. If not found,
    runs CPU inference to generate reference outputs (tokens and logits).
    Optionally saves the validation info for future use.

    Args:
        model_variant: Model identifier or path.
        max_new_tokens: Maximum number of tokens to generate.
        validation_info_outputs_dir: Directory for validation info outputs.
        save_validation_info_outputs: Whether to save validation info to disk.
        validation_model: CPU model for generating validation data.
        valid_prompt: ValidPrompt object containing prompt input IDs and extra kwargs for model execution.
        env_config: Environment configuration with attention settings and CPU dtype.
        tokenizer: HuggingFace tokenizer for the model.

    Returns:
        ValidationInfo: ValidationInfo object containing CPU reference outputs (tokens and logits)."""

    # attempt to load the cpu validation info if it is already computed
    cpu_validation_info = _load_validation_info(
        model_variant=model_variant,
        batch_size=valid_prompt.shape[0],
        seq_length=valid_prompt.shape[1],
        max_new_tokens=max_new_tokens,
        tokenizer=tokenizer,
        seed=0,
        cpu_dtype=env_config.cpu_dtype,
        attn_type=env_config.attn_type,
        validation_info_outputs_dir=validation_info_outputs_dir,
        sample_key=valid_prompt.sample_key,
    )

    if cpu_validation_info is not None:
        # Skip CPU generation if validation info is already available
        dprint(
            f"Loaded CPU validation info for program {model_variant} with prompt shape {valid_prompt} and sample key {valid_prompt.sample_key}"
        )
        return cpu_validation_info

    cpu_validation_info = extract_validation_information(
        model=validation_model,
        input_ids=valid_prompt.input_ids,
        max_new_tokens=max_new_tokens,
        post_iteration_hook=LogitsExtractorHook(),
        attn_algorithm=AttnType.MATH,
        **valid_prompt.extra_kwargs,
    )

    if not save_validation_info_outputs:
        return cpu_validation_info

    validation_info_path = get_validation_info_path(
        validation_info_dir=validation_info_outputs_dir,
        model_variant=model_variant,
        batch_size=valid_prompt.shape[0],
        seq_length=valid_prompt.shape[1],
        max_new_tokens=max_new_tokens,
        seed=0,
        attn_type=env_config.attn_type,
        dtype=env_config.cpu_dtype,
        sample_key=valid_prompt.sample_key,
    )
    cpu_validation_info.save(validation_info_path)

    return cpu_validation_info


def generate_aiu_cpu_test(
    valid_prompts: Iterable[ValidPrompt],
    model: torch.nn.Module,
    validation_model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    env_config: EnvConfig,
    model_config: DPPRunnerConfig,
    test_type: TestType,
    max_new_tokens: int,
    save_validation_info_outputs: bool,
    validation_info_outputs_dir: str,
    cross_entropy_threshold: float,
    failure_rate_threshold: float,
    timing: Timing,
    prefill_chunk_size: int,
    model_variant: str,
) -> list[Any]:
    """Generates tokens using AIU and CPU models and validates the results.

    This function iterates through prepared prompts, executes the generation
    cycle for both hardware targets, and evaluates whether the AIU outputs
    match the golden reference.

    Args:
        valid_prompts: Iterable of ValidPrompt objects containing input prompts and metadata.
        model: Compiled AIU model for inference.
        validation_model: CPU model for generating reference validation data.
        tokenizer: HuggingFace tokenizer for decoding token outputs.
        env_config: Environment configuration with attention settings.
        model_config: Model configuration with architecture details.
        test_type: Type of test being run.
        max_new_tokens: Maximum number of tokens to generate.
        save_validation_info_outputs: Whether to save CPU validation info to disk.
        validation_info_outputs_dir: Directory for saving/loading CPU validation info.
        cross_entropy_threshold: Threshold for cross-entropy difference to consider a token generation as failed.
        failure_rate_threshold: Threshold for the failure rate to consider the test case as failed.
        timing: Whether to collect timing information.
        prefill_chunk_size: Chunk size for prefill operations.
        model_variant: Model identifier or path for naming validation info files.

    Returns:
        List of failed cases with program ID, prompt shape, and failure rate."""

    failed_cases = []
    # for each program and valid prompt (batch size, sequence length)
    for valid_prompt in valid_prompts:
        valid_prompt.extra_kwargs["attn_name"] = env_config.attn_type
        valid_prompt.extra_kwargs["_kvcache_num_blocks_hint"] = model_config.num_blocks

        if local_rank == 0:
            dprint(f"*** testing program {valid_prompt.program_id} ***")
            dprint(
                f"program id: {valid_prompt.program_id}, valid prompt: {valid_prompt.shape}, input shape: {valid_prompt.input_ids.shape}"
            )

        # Generate or load CPU validation info
        cpu_validation_info = generate_cpu_validation(
            model_variant,
            max_new_tokens,
            validation_info_outputs_dir,
            save_validation_info_outputs,
            validation_model,
            valid_prompt.shape,
            valid_prompt.input_ids,
            valid_prompt.extra_kwargs,
            valid_prompt.sample_key,
            env_config.attn_type,
            env_config.cpu_dtype,
            tokenizer,
        )

        aiu_validation_info = generate_aiu_validation(
            test_type,
            max_new_tokens,
            timing,
            prefill_chunk_size,
            model,
            valid_prompt,
            cpu_validation_info=cpu_validation_info,
        )

        if test_type == TestType.METRICS:
            failure_rate = evaluate_cross_entropy_metrics(
                cross_entropy_threshold,
                aiu_validation_info,
                cpu_validation_info,
                valid_prompt.program_id,
                valid_prompt.shape,
                tokenizer,
            )
            if failure_rate > failure_rate_threshold:
                failed_cases.append(
                    (valid_prompt.program_id, valid_prompt.shape, failure_rate)
                )

        elif test_type == TestType.TOKENS:
            report_token_comparison(
                max_new_tokens,
                aiu_validation_info,
                cpu_validation_info,
                valid_prompt.program_id,
                tokenizer,
            )

        else:
            raise ValueError("test type must be one of metrics or tokens")

    return failed_cases


def generate_aiu_test(
    valid_prompts: Iterable[ValidPrompt],
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    env_config: EnvConfig,
    model_config: DPPRunnerConfig,
    test_type: TestType,
    max_new_tokens: int,
    timing: Timing,
    prefill_chunk_size: int,
) -> None:
    """Generates tokens using the AIU model and prints the outputs.

    Args:
        valid_prompts: Iterable of ValidPrompt objects containing input prompts and metadata.
        model: Compiled AIU model for inference.
        tokenizer: HuggingFace tokenizer for decoding token outputs.
        env_config: Environment configuration with attention settings.
        model_config: Model configuration with architecture details.
        test_type: Type of test being run.
        max_new_tokens: Maximum number of tokens to generate.
        timing: Whether to collect timing information.
        prefill_chunk_size: Chunk size for prefill operations."""

    # for each program and valid prompt (batch size, sequence length)
    for valid_prompt in valid_prompts:
        valid_prompt.extra_kwargs["attn_name"] = env_config.attn_type.value
        valid_prompt.extra_kwargs["_kvcache_num_blocks_hint"] = model_config.num_blocks

        if local_rank == 0:
            dprint(f"*** testing program {valid_prompt.program_id} ***")
            dprint(
                f"program id: {valid_prompt.program_id}, valid prompt: {valid_prompt.shape}, input shape: {valid_prompt.input_ids.shape}"
            )

        aiu_tokens = generate_aiu_validation(
            test_type,
            max_new_tokens,
            timing,
            prefill_chunk_size,
            model,
            valid_prompt,
        ).get_info("tokens")

        if local_rank != 0:
            return

        for sentence_idx, test_sentence in enumerate(aiu_tokens):
            tokens_prompt = [t.item() for t in test_sentence[:-max_new_tokens]]
            aiu_tokens_generated = [t.item() for t in test_sentence[-max_new_tokens:]]
            dprint(
                f"For Program {valid_prompt.program_id} in sentence {sentence_idx + 1}:"
            )
            dprint(f"Prompt:\n{tokenizer.decode(tokens_prompt)}")
            dprint(f"AIU tokens:\n{aiu_tokens_generated}")
            dprint(f"AIU output:\n{tokenizer.decode(aiu_tokens_generated)}")
