from typing import Tuple

import torch
from transformers import AutoTokenizer

from aiu_fms_testing_utils.testing.dpp.program_models import MetricResult
from aiu_fms_testing_utils.testing.validation import (
    ValidationInfo,
    capture_level_1_metrics,
    filter_failed_level_1_cases,
    top_k_loss_calculator,
)
from aiu_fms_testing_utils.utils.aiu_setup import local_rank, r0dprint


def _metric_calculator(reference_tensor: torch.Tensor, test_tensor: torch.Tensor):
    """Calculates cross-entropy and mean absolute difference between logit distributions.

    Args:
        reference_tensor: Reference logits tensor from CPU validation.
        test_tensor: Test logits tensor from AIU inference.

    Returns:
        MetricResult: A named tuple containing the calculated metrics."""

    cross_entropy_loss = torch.nn.CrossEntropyLoss()(
        reference_tensor, test_tensor.softmax(dim=1).to(dtype=torch.float32)
    )
    mean_abs_diff = torch.mean(
        torch.abs(
            reference_tensor.softmax(dim=1).to(dtype=torch.float32)
            - test_tensor.softmax(dim=1).to(dtype=torch.float32)
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
        float: Failure rate (number of failed tokens / total tokens)."""

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
            r0dprint(
                f'For Program {program_id} in sentence {sentence_idx + 1}: the metric for token {token_idx} is {metrics_value}, AIU ID="{aiu_token.item()}" | STR="{aiu_str}" -- CPU ID="{cpu_token.item()}" | CPU STR="{cpu_str}"'
            )

    ce_fail_responses = filter_failed_level_1_cases(
        level_1_metrics, lambda m: m[0] >= cross_entropy_threshold
    )
    failure_rate = len(ce_fail_responses) / len(level_1_metrics)

    return failure_rate


def evaluate_token_accuracy(
    max_new_tokens: int,
    aiu_validation_info: ValidationInfo,
    cpu_validation_info: ValidationInfo,
    program_id: str,
    tokenizer: AutoTokenizer,
) -> float:
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

    Returns:
        float: Failure rate (percentage of mismatched tokens)."""

    total_mismatches = 0
    total_tokens = 0
    for sentence_idx, (reference_sentence, test_sentence) in enumerate(
        zip(
            cpu_validation_info.get_info("tokens"),
            aiu_validation_info.get_info("tokens"),
        )
    ):
        tokens_prompt = reference_sentence[:-max_new_tokens]
        cpu_tokens_generated = reference_sentence[-max_new_tokens:]
        aiu_tokens_generated = test_sentence[-max_new_tokens:]

        # Remove leading padding tokens using torch operations
        pad_mask = tokens_prompt != tokenizer.pad_token_id
        first_non_pad = pad_mask.nonzero(as_tuple=True)[0]
        if len(first_non_pad) > 0:
            tokens_prompt_without_pad = tokens_prompt[first_non_pad[0] :]
        else:
            tokens_prompt_without_pad = tokens_prompt
        prompt_length = tokens_prompt_without_pad.size(0)

        # Calculate token mismatch failure rate
        num_mismatches = (cpu_tokens_generated != aiu_tokens_generated).sum().item()
        prompt_tokens = cpu_tokens_generated.size(0)
        total_mismatches += num_mismatches
        total_tokens += prompt_tokens

        r0dprint(f"Prompt Length: {prompt_length}")
        r0dprint(f"For Program {program_id} in sentence {sentence_idx + 1}:")
        r0dprint(f"Prompt:\n{tokenizer.decode(tokens_prompt_without_pad)}")
        r0dprint(f"CPU tokens:\n{cpu_tokens_generated}")
        r0dprint(f"AIU tokens:\n{aiu_tokens_generated}")
        r0dprint(f"CPU output:\n{tokenizer.decode(cpu_tokens_generated)}")
        r0dprint(f"AIU output:\n{tokenizer.decode(aiu_tokens_generated)}")

    failure_rate = total_mismatches / total_tokens if total_tokens > 0 else 0.0
    r0dprint(
        f"Token Failure Rate: {failure_rate:.2%} ({num_mismatches}/{prompt_tokens} mismatches)"
    )
    return failure_rate
