import itertools
import json
import os
import random

from huggingface_hub import hf_hub_download
from aiu_fms_testing_utils.testing.dpp.prepare_programs import get_programs_to_test
from aiu_fms_testing_utils.testing.dpp.program_models import (
    PreparedInputs,
    ProgramInfo,
    ValidPrompt,
)
from aiu_fms_testing_utils.utils.aiu_setup import dprint, r0dprint
from fms.utils.generation import pad_input_ids


import torch
from transformers import AutoTokenizer


import time
from typing import List, Optional, Tuple, Callable, Generator

from aiu_fms_testing_utils.utils.paged import ProgramCriteria, get_programs_prompts
from aiu_fms_testing_utils.testing.dpp.constants import PAD_MULTIPLE


SHARE_GPT_DATASET = (
    "anon8231489123/ShareGPT_Vicuna_unfiltered",
    "ShareGPT_V3_unfiltered_cleaned_split.json",
)
RAG_FACTOID_DATASET = ("", "")


def _prepare_inputs(
    batch_size: int,
    seq_length: int,
    tokenizer: AutoTokenizer,
    sampler: Callable[..., tuple[list[tuple[str, int]], str]],
    dataset_path: str,
    allow_truncation: bool,
    enforce_sizes: List[int] = [],
    seed: int = 0,
) -> PreparedInputs:
    """Prepares and tokenizes input prompts for model inference.

    Samples prompts from a dataset using the provided sampler, tokenizes them,
    and pads them to the specified sequence length. Handles cases where fewer
    prompts are available than requested by repeating the first prompt.

    Args:
        batch_size: Number of prompts to sample for the batch.
        seq_length: Target sequence length for padding.
        tokenizer: HuggingFace tokenizer for encoding prompts.
        sampler: Callable that samples prompts from the dataset.
        dataset_path: Path to the dataset file.
        allow_truncation: If True, allows truncating prompts longer than seq_length.
        enforce_sizes: List of specific sequence lengths to enforce for sampling.
        seed: Random seed for reproducible sampling.

    Returns:
        Tuple containing:
            - input_ids: Padded tensor of tokenized input IDs with shape (batch_size, seq_length).
            - extra_kwargs: Dictionary with additional model inputs including attention mask.
            - sample_key: String identifier for the sampled prompts.

    Raises:
        ValueError: If no valid prompts exist in the dataset for the requested shape."""

    start = time.time()
    prompts_and_sizes, sample_key = sampler(
        dataset_path,
        batch_size,
        tokenizer,
        32,
        seq_length * 2 if allow_truncation else seq_length,
        seed,
        enforce_sizes=enforce_sizes,
        truncation=allow_truncation,
        return_key=True,
    )
    end = time.time()

    r0dprint(
        f"Extracted {len(prompts_and_sizes)} prompts in {(end - start):.4f} seconds"
    )

    prompt_list = []
    for prompt, size in prompts_and_sizes:
        encoded = tokenizer.encode(prompt, return_tensors="pt").squeeze(0)
        if size > seq_length:
            assert allow_truncation
            encoded = encoded[:seq_length]
        prompt_list.append(encoded)

    if not prompt_list:
        raise ValueError(
            f"No valid prompt sample exists in dataset for input shape (Batch Size={batch_size}, Seq Length={seq_length})"
        )
    if len(prompt_list) < batch_size:
        dprint(
            f"You requested {batch_size} prompts but we were only able to get {len(prompt_list)} valid prompts. We will be repeating the first prompt."
        )
        prompt_list = [prompt_list[0]] * (batch_size - len(prompt_list)) + prompt_list

    input_ids, extra_kwargs = pad_input_ids(prompt_list, min_pad_length=seq_length)
    extra_kwargs["mask"] = extra_kwargs["mask"].to(torch.float16)

    return PreparedInputs(
        input_ids=input_ids, extra_kwargs=extra_kwargs, sample_key=sample_key
    )


def _get_valid_prompts_by_custom_shape(
    program_map: dict,
    custom_shape: Tuple[int, int],
    tokenizer: AutoTokenizer,
    sampler: Callable[..., tuple[list[tuple[str, int]], str]],
    dataset_path: str,
    allow_truncation: bool,
) -> Generator[ValidPrompt, None, None]:
    """Selects prompts matching a custom shape for user-provided datasets.

    Args:
        program_map: Dictionary mapping program sequences to valid prompt shapes.
        custom_shape: Tuple of (batch_size, seq_length) specified by the user for custom datasets.
        tokenizer: HuggingFace tokenizer for encoding prompts.
        sampler: Callable for sampling prompts from the dataset.
        dataset_path: Path to the dataset for sampling prompts.
        allow_truncation: If True, allows truncating prompts that exceed max sequence length.
    Yields:
        ValidPrompt: A named tuple for prompts matching the custom shape."""

    found_valid_shape = False
    for program_criteria_seq, valid_prompt_shapes in program_map.items():
        for valid_prompt_shape in valid_prompt_shapes:
            if valid_prompt_shape != custom_shape:
                continue

            input_ids, extra_kwargs, sample_key = _prepare_inputs(
                valid_prompt_shape[0],
                valid_prompt_shape[1],
                tokenizer,
                sampler,
                dataset_path,
                allow_truncation,
                enforce_sizes=[valid_prompt_shape[1]],
            )
            yield ValidPrompt(
                program_id=program_criteria_seq[0].program_id,
                shape=custom_shape,
                input_ids=input_ids,
                extra_kwargs=extra_kwargs,
                sample_key=sample_key,
            )
            found_valid_shape = True
            break

        if found_valid_shape:
            break

    if not found_valid_shape:
        r0dprint(
            f"No valid prompt shape was found which would result in program {program_criteria_seq[0].program_id} that satisfied the custom shape {custom_shape}"
        )


def _get_valid_prompts_by_shape(
    program_map: dict,
    program_info: ProgramInfo,
    tokenizer: AutoTokenizer,
    sampler: Callable[..., tuple[list[tuple[str, int]], str]],
    dataset_path: str,
    allow_truncation: bool,
    pad_multiple: int,
    enforce_homogeneous_prompt_programs: bool,
) -> Generator[ValidPrompt, None, None]:
    """Selects valid prompts matching program criteria and constraints.

    Args:
        program_map: Dictionary mapping program sequences to valid prompt shapes.
        program_info: ProgramInfo object specifying the program and its constraints.
        tokenizer: HuggingFace tokenizer for encoding prompts.
        sampler: Callable for sampling prompts from the dataset.
        dataset_path: Path to the dataset for sampling prompts.
        allow_truncation: If True, allows truncating prompts that exceed max sequence length.
        pad_multiple: Padding granularity for sequence lengths (typically 64).
        enforce_homogeneous_prompt_programs: If True, ensures all prompts in a batch use the same decode program.
    Yields:
        ValidPrompt: A named tuple matching the program criteria and constraints for testing."""

    used_keys = set()
    # for each program, we need to check if we have a shape that satisfies the --programs request
    for program_criteria_seq, valid_prompt_shapes in program_map.items():
        # if ? or numeric => we need to check if we have found at least one valid key to stop
        if (
            program_info.program_id == "?" or program_info.program_id.isnumeric()
        ) and len(used_keys) > 0:
            break
        # if * => we need to see if we have found the first key to see if we should skip
        elif program_info.program_id == "*" and program_criteria_seq[0] in used_keys:
            continue

        for valid_prompt_shape in valid_prompt_shapes:
            # make sure the criteria for batch limit and prompt limit is satisfied
            # eval is safe here because we have limited what type and limit can be before

            batch_check = eval(
                f"valid_prompt_shape[0] {program_info.batch_size_limit_type} {program_info.batch_size_limit}"
            )
            prompt_check = eval(
                f"valid_prompt_shape[1] {program_info.prompt_length_limit_type} {program_info.prompt_length_limit}"
            )
            if not batch_check or not prompt_check:
                continue

            # when we enforce homogeneous prompt programs, we will cycle through all sizes between the min of a program and the valid prompt sequence length
            # if there does not exist enough sequence sizes between this range, we will cycle back to the beginning
            # in the event we don't have enough sequences that satisfy the enforce_sizes, we will repeat sequences and warn the user
            enforce_sizes = [valid_prompt_shape[1]]
            if enforce_homogeneous_prompt_programs:
                # this will get the number of bits for the sequence length and shift to get the power of 2 that is less than or equal to the sequence length
                tkv_cutoff = 1 << (valid_prompt_shape[1].bit_length() - 1)
                possible_seq_lengths = [
                    _ for _ in range(tkv_cutoff, valid_prompt_shape[1], pad_multiple)
                ]
                # favor sequences that are close to the valid prompt length
                possible_seq_lengths.reverse()
                enforce_sizes = enforce_sizes + list(
                    itertools.islice(
                        itertools.cycle(possible_seq_lengths),
                        valid_prompt_shape[0] - 1,
                    )
                )

            try:
                input_ids, extra_kwargs, sample_key = _prepare_inputs(
                    batch_size=valid_prompt_shape[0],
                    seq_length=valid_prompt_shape[1],
                    tokenizer=tokenizer,
                    sampler=sampler,
                    dataset_path=dataset_path,
                    allow_truncation=allow_truncation,
                    enforce_sizes=enforce_sizes,
                )
                yield ValidPrompt(
                    program_id=program_criteria_seq[0],
                    shape=valid_prompt_shape,
                    input_ids=input_ids,
                    extra_kwargs=extra_kwargs,
                    sample_key=sample_key,
                )
                break
            except ValueError as e:
                dprint(f"Failed to prepare inputs for shape {valid_prompt_shape}: {e}")

    if len(used_keys) == 0:
        r0dprint(
            f"No valid prompt shape was found which would result in program {program_info.program_id} that satisfied batch{program_info.batch_size_limit_type}{program_info.batch_size_limit} and prompt_length{program_info.prompt_length_limit_type}{program_info.prompt_length_limit}"
        )


def _get_valid_prompts(
    program_map: dict,
    dataset_path: str,
    enforce_homogeneous_prompt_programs: bool,
    programs_to_test: List[ProgramInfo],
    program_criteria_list: List[ProgramCriteria],
    tokenizer: AutoTokenizer,
    sampler: Callable[..., tuple[list[tuple[str, int]], str]],
    allow_truncation: bool,
    pad_multiple: int,
) -> Generator[ValidPrompt, None, None]:
    """Generator that yields valid prompts matching program criteria and constraints.

    Iterates through programs to test and finds prompts from the dataset that satisfy
    the program's batch size and prompt length constraints. For custom datasets, uses
    the provided shape directly. For other datasets, samples prompts matching the
    program criteria. When enforce_homogeneous_prompt_programs is True, generates
    multiple sequence lengths within a batch to ensure all prompts hit the same program.

    Args:
        program_map: Dictionary mapping program sequences to valid prompt shapes.
        dataset_path: Path to the dataset for sampling prompts.
        enforce_homogeneous_prompt_programs: If True, ensures all prompts in a batch
                                            use the same decode program.
        programs_to_test: List of ProgramInfo objects specifying programs and constraints.
        program_criteria_list: List of ProgramCriteria defining program boundaries.
        tokenizer: HuggingFace tokenizer for encoding prompts.
        sampler: Callable for sampling prompts from the dataset.
        allow_truncation: If True, allows truncating prompts exceeding max length.
        pad_multiple: Padding granularity for sequence lengths (typically 64).

    Yields:
        ValidPrompt: A named tuple matching the program criteria and constraints for testing."""

    for program_info in programs_to_test:
        program_id = program_info.program_id

        filtered_program_map = program_map
        if program_id.isnumeric():
            filtered_program_map = {
                k: v
                for k, v in program_map.items()
                if k[0] == program_criteria_list[int(program_id)]
            }

        yield from _get_valid_prompts_by_shape(
            filtered_program_map,
            program_info,
            tokenizer,
            sampler,
            dataset_path,
            allow_truncation,
            pad_multiple,
            enforce_homogeneous_prompt_programs,
        )


def prepare_test_prompts(
    program_criteria_json_path: str,
    programs: List[str],
    max_new_tokens: int,
    prioritize_large_batch_sizes: bool,
    enforce_homogeneous_prompt_programs: bool,
    max_batch_size: int,
    max_tkv: int,
    tkv_limit: Optional[int],
    tokenizer: AutoTokenizer,
    sampler: Callable[..., tuple[list[tuple[str, int]], str]],
    allow_truncation: bool,
    custom_shape: Optional[Tuple[int, int]],
    dataset_path: str,
) -> Generator[ValidPrompt, None, None]:
    """Parses program criteria and generates the sequence of valid test prompts.

    Args:
        program_criteria_json_path: Path to JSON file containing program criteria definitions.
        programs: List of program specifications from command line arguments.
        max_new_tokens: Maximum number of tokens to generate for each prompt.
        prioritize_large_batch_sizes: If True, prioritizes larger batch sizes when selecting prompts.
        enforce_homogeneous_prompt_programs: If True, ensures all prompts in a batch use the same decode program.
        max_batch_size: Maximum batch size to consider when selecting prompts.
        max_tkv: Maximum total key-value size to consider when selecting prompts.
        tkv_limit: Optional limit on total key-value size for prompts.
        tokenizer: HuggingFace tokenizer for encoding prompts.
        sampler: Callable for sampling prompts from the dataset.
        allow_truncation: If True, allows truncating prompts that exceed max sequence length.
        custom_shape: Optional tuple of (batch_size, seq_length) for custom datasets.
        dataset_path: Path to the dataset for sampling prompts."""

    with open(program_criteria_json_path, "r") as f:
        program_criteria_json_list = json.load(f)["programs"]
        program_criteria_list = []
        for i, d in enumerate(program_criteria_json_list):
            program_criteria_list.append(
                ProgramCriteria(
                    i,
                    d["max_batch"],
                    d["max_tkv"],
                    d["batch_granularity"],
                    d["tkv_granularity"],
                )
            )

        programs_to_test = get_programs_to_test(programs, program_criteria_list)

    # FIXME: filter condition for this on prompt and batch
    program_map = get_programs_prompts(
        program_criteria_list=program_criteria_list,
        multiple=PAD_MULTIPLE,
        max_batch_size=max_batch_size,
        max_tkv=max_tkv,
        program_cycles=max_new_tokens,
        tkv_limit=tkv_limit,
        prioritize_large_batch_sizes=prioritize_large_batch_sizes,
    )

    for v in program_map.values():
        random.Random(42).shuffle(v)

    if custom_shape:
        # Exit early if the user has selected a custom shape
        return _get_valid_prompts_by_custom_shape(
            program_map,
            custom_shape,
            tokenizer,
            sampler,
            dataset_path,
            allow_truncation,
        )

    # Select concrete prompts and program associations
    return _get_valid_prompts(
        program_map=program_map,
        dataset_path=dataset_path,
        enforce_homogeneous_prompt_programs=enforce_homogeneous_prompt_programs,
        programs_to_test=programs_to_test,
        program_criteria_list=program_criteria_list,
        tokenizer=tokenizer,
        sampler=sampler,
        allow_truncation=allow_truncation,
        pad_multiple=PAD_MULTIPLE,
    )


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
        r0dprint("Using ShareGPT dataset from HuggingFace")
        dataset_type = "sharegpt"
        # Fetch from HuggingFace
        local_dataset_path = hf_hub_download(
            repo_id=SHARE_GPT_DATASET[0],
            filename=SHARE_GPT_DATASET[1],
            repo_type="dataset",
        )
    elif dataset_path == "rag_factoid":
        r0dprint("Using RAG Factoid dataset from HuggingFace")
        dataset_type = "rag_factoid"
        # Fetch from HuggingFace
        local_dataset_path = hf_hub_download(
            repo_id=RAG_FACTOID_DATASET[0],
            filename=RAG_FACTOID_DATASET[1],
            repo_type="dataset",
        )
    elif dataset_path is None:
        r0dprint(f"Using a custom dataset at {dataset_path}")
        dataset_type = "custom"
        local_dataset_path = dataset_path
    else:
        raise ValueError(f"Unsupported dataset_path: {dataset_path}")

    if not os.path.exists(local_dataset_path):
        raise FileNotFoundError(f"Dataset file not found at {local_dataset_path}")

    return dataset_type, local_dataset_path
