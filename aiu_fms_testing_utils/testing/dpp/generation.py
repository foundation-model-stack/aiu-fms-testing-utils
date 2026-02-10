from aiu_fms_testing_utils.testing.dpp.metrics_validation import evaluate_cross_entropy_metrics, report_token_comparison
from aiu_fms_testing_utils.testing.dpp.program_models import EnvConfig, ValidPrompt
from aiu_fms_testing_utils.testing.validation import GoldenTokenHook, LogitsExtractorHook, ValidationInfo, extract_validation_information, get_validation_info_path
from aiu_fms_testing_utils.utils.aiu_setup import dprint, local_rank
from aiu_fms_testing_utils.utils.dpp_config import DPPRunnerConfig
from aiu_fms_testing_utils.testing.dpp.metrics_validation import _load_validation_info


import torch
from transformers import AutoTokenizer


from typing import Any, Dict, Iterable, Optional


def generate_aiu_validation(
    test_type: str,
    max_new_tokens: int,
    timing: str,
    prefill_chunk_size: int,
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    cpu_validation_info: Optional[ValidationInfo],
    extra_kwargs: Dict[str, Any],
) -> ValidationInfo:
    """Generates AIU validation information by running inference on the compiled model.

    Executes the AIU-compiled model to generate tokens and extract logits. If CPU
    validation info is available and test_type is "metrics", injects golden tokens
    from CPU validation to ensure consistent decode paths for metric comparison.

    Args:
        test_type: Type of test being run ("metrics" or "tokens").
        max_new_tokens: Maximum number of tokens to generate.
        timing: Whether to collect timing information.
        prefill_chunk_size: Chunk size for prefill operations.
        model: Compiled AIU model for inference.
        input_ids: Tokenized input tensor.
        cpu_validation_info: Optional CPU validation data for golden token injection.
        extra_kwargs: Dictionary with attention mask and other model inputs.

    Returns:
        ValidationInfo: ValidationInfo object containing AIU outputs (tokens, logits,
        and optional timing information).
    """
    golden_hook = None
    if test_type == "metrics" and cpu_validation_info:
        golden_hook = GoldenTokenHook(cpu_validation_info.get_info("tokens"))

    aiu_validation_info = extract_validation_information(
        model=model,
        input_ids=input_ids,
        max_new_tokens=max_new_tokens,
        post_iteration_hook=golden_hook,
        last_n_tokens=64,
        timing=timing,
        prefill_chunk_size=prefill_chunk_size,
        **extra_kwargs,
    )

    return aiu_validation_info


def generate_cpu_validation(
    model_variant: str,
    max_new_tokens: int,
    validation_info_outputs_dir: str,
    save_validation_info_outputs: bool,
    validation_model: torch.nn.Module,
    valid_prompt,
    input_ids: torch.Tensor,
    extra_kwargs: Dict[str, Any],
    sample_key: str,
    attn_name: str,
    cpu_dtype: str,
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
        valid_prompt: Tuple of (batch_size, seq_length) for the prompt shape.
        input_ids: Tokenized input tensor.
        extra_kwargs: Dictionary with attention mask and other model inputs.
        sample_key: String identifier for the sampled prompts.
        attn_name: Name of the attention algorithm used.
        cpu_dtype: Data type string for CPU validation ("fp8" or "fp32").
        tokenizer: HuggingFace tokenizer for the model.

    Returns:
        ValidationInfo: ValidationInfo object containing CPU reference outputs
        (tokens and logits).
    """
    # attempt to load the cpu validation info if it is already computed
    cpu_validation_info = _load_validation_info(
        model_variant=model_variant,
        batch_size=valid_prompt[0],
        seq_length=valid_prompt[1],
        max_new_tokens=max_new_tokens,
        tokenizer=tokenizer,
        seed=0,
        cpu_dtype=cpu_dtype,
        attn_type=attn_name,
        validation_info_outputs_dir=validation_info_outputs_dir,
        sample_key=sample_key,
    )
    if cpu_validation_info is None:
        cpu_validation_info = extract_validation_information(
            model=validation_model,
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            post_iteration_hook=LogitsExtractorHook(),
            attn_algorithm="math",
            **extra_kwargs,
        )
        if save_validation_info_outputs:
            cpu_validation_info.save(
                get_validation_info_path(
                    validation_info_dir=validation_info_outputs_dir,
                    model_variant=model_variant,
                    batch_size=valid_prompt[0],
                    seq_length=valid_prompt[1],
                    max_new_tokens=max_new_tokens,
                    seed=0,
                    attn_type=attn_name,
                    dtype=cpu_dtype,
                    sample_key=sample_key,
                )
            )

    return cpu_validation_info


def generate_validation_info_and_test(
    valid_prompts: Iterable[ValidPrompt],
    model: torch.nn.Module,
    validation_model: Optional[torch.nn.Module],
    tokenizer: AutoTokenizer,
    env_config: EnvConfig,
    model_config: DPPRunnerConfig,
    test_type: str,
    max_new_tokens: int,
    skip_validation: bool,
    save_validation_info_outputs: bool,
    validation_info_outputs_dir: str,
    cross_entropy_threshold: float,
    failure_rate_threshold: float,
    timing: str,
    prefill_chunk_size: int,
    model_variant: str,
) -> list[Any]:
    """Generates tokens using AIU and CPU models and validates the results.

    This function iterates through prepared prompts, executes the generation
    cycle for both hardware targets, and evaluates whether the AIU outputs
    match the golden reference.
    """

    failed_cases = []
    # for each program and valid prompt (batch size, sequence length)
    for valid_prompt in valid_prompts:
        valid_prompt.extra_kwargs["attn_name"] = env_config.attn_name
        valid_prompt.extra_kwargs["_kvcache_num_blocks_hint"] = model_config.num_blocks

        if local_rank == 0:
            dprint(f"*** testing program {valid_prompt.program_id} ***")
            dprint(
                f"program id: {valid_prompt.program_id}, valid prompt: {valid_prompt.shape}, input shape: {valid_prompt.input_ids.shape}"
            )

        if not skip_validation:
            # Generate or load CPU validation info
            cpu_validation_info = generate_cpu_validation(
                model_variant=model_variant,
                max_new_tokens=max_new_tokens,
                validation_info_outputs_dir=validation_info_outputs_dir,
                save_validation_info_outputs=save_validation_info_outputs,
                validation_model=validation_model,
                valid_prompt=valid_prompt.shape,
                input_ids=valid_prompt.input_ids,
                extra_kwargs=valid_prompt.extra_kwargs,
                sample_key=valid_prompt.sample_key,
                attn_name=env_config.attn_name,
                cpu_dtype=env_config.cpu_dtype,
                tokenizer=tokenizer,
            )

            aiu_validation_info = generate_aiu_validation(
                test_type=test_type,
                max_new_tokens=max_new_tokens,
                timing=timing,
                prefill_chunk_size=prefill_chunk_size,
                model=model,
                input_ids=valid_prompt.input_ids,
                cpu_validation_info=cpu_validation_info,
                extra_kwargs=valid_prompt.extra_kwargs,
            )

            if test_type == "metrics":
                failure_rate = evaluate_cross_entropy_metrics(
                    cross_entropy_threshold=cross_entropy_threshold,
                    aiu_validation_info=aiu_validation_info,
                    cpu_validation_info=cpu_validation_info,
                    program_id=valid_prompt.program_id,
                    prompt_shape=valid_prompt.shape,
                    tokenizer=tokenizer,
                )
                if failure_rate > failure_rate_threshold:
                    failed_cases.append(
                        (valid_prompt.program_id, valid_prompt.shape, failure_rate)
                    )

            elif test_type == "tokens":
                report_token_comparison(
                    max_new_tokens=max_new_tokens,
                    aiu_validation_info=aiu_validation_info,
                    cpu_validation_info=cpu_validation_info,
                    program_id=valid_prompt.program_id,
                    tokenizer=tokenizer,
                )

            else:
                raise ValueError("test type must be one of metrics or tokens")
        else:
            aiu_validation_info = generate_aiu_validation(
                test_type=test_type,
                max_new_tokens=max_new_tokens,
                timing=timing,
                prefill_chunk_size=prefill_chunk_size,
                model=model,
                input_ids=valid_prompt.input_ids,
                cpu_validation_info=None,
                extra_kwargs=valid_prompt.extra_kwargs,
            )

            if local_rank == 0:
                for sentence_idx, test_sentence in enumerate(
                    aiu_validation_info.get_info("tokens")
                ):
                    tokens_prompt = [t.item() for t in test_sentence[:-max_new_tokens]]
                    aiu_tokens_generated = [
                        t.item() for t in test_sentence[-max_new_tokens:]
                    ]
                    dprint(
                        f"For Program {valid_prompt.program_id} in sentence {sentence_idx + 1}:"
                    )
                    dprint(f"Prompt:\n{tokenizer.decode(tokens_prompt)}")
                    dprint(f"AIU tokens:\n{aiu_tokens_generated}")
                    dprint(f"AIU output:\n{tokenizer.decode(aiu_tokens_generated)}")

    return failed_cases
