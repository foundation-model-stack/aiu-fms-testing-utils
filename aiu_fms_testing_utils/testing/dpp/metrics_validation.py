from itertools import dropwhile
from aiu_fms_testing_utils.testing.dpp.program_models import MetricResult
from aiu_fms_testing_utils.testing.validation import (
    ValidationInfo,
    capture_level_1_metrics,
    filter_failed_level_1_cases,
    find_validation_info_path,
    load_validation_information,
    top_k_loss_calculator,
)
from aiu_fms_testing_utils.utils.aiu_setup import dprint, local_rank


import torch
from transformers import AutoTokenizer


from typing import Tuple


def _metric_calculator(r: torch.Tensor, t: torch.Tensor):
    """Calculates cross-entropy and mean absolute difference between logit distributions.

    Args:
        r: Reference logits tensor from CPU validation.
        t: Test logits tensor from AIU inference.

    Returns:
        MetricResult: A named tuple containing the calculated metrics.
    """
    cross_entropy_loss = torch.nn.CrossEntropyLoss()(
        r, t.softmax(dim=1).to(dtype=torch.float32)
    )
    mean_abs_diff = torch.mean(
        torch.abs(
            r.softmax(dim=1).to(dtype=torch.float32)
            - t.softmax(dim=1).to(dtype=torch.float32)
        )
    )
    return MetricResult(
        cross_entropy_loss=cross_entropy_loss.item(), mean_abs_diff=mean_abs_diff.item()
    )


def evaluate_cross_entropy_metrics(
    cross_entropy_threshold: float,
    aiu_validation_info: ValidationInfo,
    cpu_validation_info: ValidationInfo,
    program_id: str,
    prompt_shape: Tuple[int, int],
    tokenizer: AutoTokenizer,
) -> float:
    """Evaluates cross-entropy metrics between AIU and CPU outputs.

    Computes cross-entropy and mean difference metrics between AIU and CPU logits
    for each generated token. Prints detailed comparison including token IDs and
    decoded strings. Calculates failure rate based on cross-entropy threshold.

    Args:
        cross_entropy_threshold: Maximum acceptable cross-entropy for a passing token.
        aiu_validation_info: ValidationInfo from AIU inference.
        cpu_validation_info: ValidationInfo from CPU reference.
        program_id: ID of the program being tested.
        prompt_shape: Tuple of (batch_size, seq_length).
        tokenizer: HuggingFace tokenizer for decoding tokens.

    Returns:
        float: Failure rate (number of failed tokens / total tokens).
    """
    level_1_metrics = capture_level_1_metrics(
        cpu_validation_info.get_info("logits"),
        aiu_validation_info.get_info("logits"),
        top_k_loss_calculator(20, _metric_calculator),
    )

    if local_rank == 0:
        cpu_tokens = cpu_validation_info.get_info("tokens")
        for sentence_idx, token_idx, metrics_value in level_1_metrics:
            aiu_token = torch.argmax(
                aiu_validation_info.get_info("logits")[sentence_idx][token_idx], dim=-1
            )
            cpu_token = cpu_tokens[sentence_idx][prompt_shape[1] + token_idx]
            aiu_str = tokenizer.decode(aiu_token).replace(
                "\n", "<NEWLINE>"
            )  # remove newlines for readability
            cpu_str = tokenizer.decode(cpu_token).replace(
                "\n", "<NEWLINE>"
            )  # remove newlines for readability
            dprint(
                f'For Program {program_id} in sentence {sentence_idx + 1}: the metric for token {token_idx} is {metrics_value}, AIU ID="{aiu_token.item()}" | STR="{aiu_str}" -- CPU ID="{cpu_token.item()}" | CPU STR="{cpu_str}"'
            )

    ce_fail_responses = filter_failed_level_1_cases(
        level_1_metrics, lambda m: m[0] >= cross_entropy_threshold
    )
    failure_rate = len(ce_fail_responses) / len(level_1_metrics)

    return failure_rate


def report_token_comparison(
    max_new_tokens: int,
    aiu_validation_info: ValidationInfo,
    cpu_validation_info: ValidationInfo,
    program_id: str,
    tokenizer: AutoTokenizer,
) -> None:
    """Reports side-by-side comparison of AIU and CPU generated token sequences.

    Prints detailed comparison of generated tokens between AIU and CPU models,
    including the original prompt, token IDs, and decoded text. Only executes
    on rank 0 in distributed settings. Used for qualitative analysis of model
    outputs rather than quantitative metrics.

    Args:
        max_new_tokens: Number of tokens generated after the prompt.
        aiu_validation_info: ValidationInfo from AIU inference.
        cpu_validation_info: ValidationInfo from CPU reference.
        program_id: ID of the program being tested.
        tokenizer: HuggingFace tokenizer for decoding tokens.
    """
    if local_rank != 0:
        return

    for sentence_idx, (reference_sentence, test_sentence) in enumerate(
        zip(
            cpu_validation_info.get_info("tokens"),
            aiu_validation_info.get_info("tokens"),
        )
    ):
        tokens_prompt = [t.item() for t in reference_sentence[:-max_new_tokens]]
        cpu_tokens_generated = [t.item() for t in reference_sentence[-max_new_tokens:]]
        aiu_tokens_generated = [t.item() for t in test_sentence[-max_new_tokens:]]
        tokens_prompt_without_pad = list(
            dropwhile(lambda x: x == tokenizer.pad_token_id, tokens_prompt)
        )
        prompt_length = len([token_id for token_id in tokens_prompt_without_pad])
        dprint(f"Prompt Length: {prompt_length}")
        dprint(f"For Program {program_id} in sentence {sentence_idx + 1}:")
        dprint(f"Prompt:\n{tokenizer.decode(tokens_prompt_without_pad)}")
        dprint(f"CPU tokens:\n{cpu_tokens_generated}")
        dprint(f"AIU tokens:\n{aiu_tokens_generated}")
        dprint(f"CPU output:\n{tokenizer.decode(cpu_tokens_generated)}")
        dprint(f"AIU output:\n{tokenizer.decode(aiu_tokens_generated)}")


def _load_validation_info(
    model_variant,
    batch_size,
    seq_length,
    max_new_tokens,
    tokenizer,
    seed,
    cpu_dtype: str,
    attn_type: str,
    validation_info_outputs_dir: str,
    sample_key: str | None = None,
) -> ValidationInfo | None:
    """Loads pre-computed CPU validation information from disk if available.

    Searches for a previously saved validation info file matching the specified
    parameters. If found, loads and returns the validation information to avoid
    redundant CPU computation.

    Args:
        model_variant: Model identifier or path (HuggingFace format).
        batch_size: Batch size used for validation.
        seq_length: Sequence length used for validation.
        max_new_tokens: Number of tokens to generate during validation.
        tokenizer: HuggingFace tokenizer for the model.
        seed: Random seed used for validation.
        cpu_dtype: Data type string for CPU validation ("fp8" or "fp32").
        attn_type: Attention algorithm type used.
        validation_info_outputs_dir: Directory containing saved validation outputs.
        sample_key: Optional identifier for the specific prompt sample used.

    Returns:
        ValidationInfo object if a matching file is found, None otherwise."""

    full_path = find_validation_info_path(
        validation_info_dir=validation_info_outputs_dir,
        model_variant=model_variant,
        batch_size=batch_size,
        seq_length=seq_length,
        max_new_tokens=max_new_tokens,
        seed=seed,
        attn_type=attn_type,
        version_allow_decrement=True,
        dtype=cpu_dtype,
        sample_key=sample_key,
    )

    if full_path is not None:
        dprint(f"cpu validation info found for seed={seed} -- loading it")
        return load_validation_information(full_path, "logits", batch_size, tokenizer)

    return None
