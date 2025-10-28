# Standard
from typing import Optional, List, Tuple
import json
import os
import random
import requests
import time
import bisect

# Third Party

from aiu_fms_testing_utils.utils.aiu_setup import dprint, rank, world_size
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from aiu_fms_testing_utils.testing.utils import format_kwargs_to_string

from fms.utils.generation import pad_input_ids
import torch
import torch.nn as nn
import math
import contextlib
import warnings


@contextlib.contextmanager
def stagger_region(limit: int):
    """
    Limit the number of concurrent processes into this region of code.
    Processes yield from this function when they are allowed to enter the region of code.
    Processes return from this function when all of the processes have completed the region of code.

    :param limit: Number of concurrent processes allowed in the code region if > 0.
    """
    if limit > 0 and limit != world_size:
        for _set in range(math.ceil(world_size / float(limit))):
            if rank < (_set + 1) * limit:
                break
            torch.distributed.barrier()
        dprint(
            f"Stagger: Enter (Set: {_set + 1} of {math.ceil(world_size / float(limit))})"
        )
    yield
    if limit > 0 and limit != world_size:
        for _set in range(math.ceil(world_size / float(limit))):
            if rank >= (_set + 1) * limit:
                continue
            torch.distributed.barrier()
        dprint("Stagger: All Complete")


def warmup_model(
    model: nn.Module,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    compile_dynamic_sendnn: bool = False,
    use_cache: bool = True,
    stagger_update_lazyhandle: int = 0,
    prefill_chunk_size: int = 0,
    **extra_kwargs,
):
    import torch_sendnn

    attention_specific_kwargs = {}
    attn_name = extra_kwargs.get("attn_name", "sdpa")
    if "paged" in attn_name:
        from aiu_fms_testing_utils.utils.paged import generate, adjust_inputs_to_batch

        attention_specific_kwargs["prefill_chunk_size"] = prefill_chunk_size
    else:
        # TODO: Add a unified generation dependent on attn_type
        from fms.utils.generation import generate

        attention_specific_kwargs["contiguous_cache"] = True
        attention_specific_kwargs["max_seq_len"] = input_ids.shape[1] + max_new_tokens

    dprint("AIU warmup")
    pt_compile_model_time = time.time()

    # adjust inputs depending on attn_type and dynamic shapes
    _warmup_input_ids = input_ids
    _extra_kwargs = extra_kwargs
    _max_new_tokens = max_new_tokens
    if compile_dynamic_sendnn:
        _max_new_tokens = 2
        # When performing fp8 paged attention, we must pad to batch size 2
        # this is fixed in torch >= 2.8
        if attn_name == "spyre_paged_attn_fp8":
            _warmup_input_ids, _extra_kwargs = adjust_inputs_to_batch(
                input_ids,
                **extra_kwargs,
            )

    extra_kwargs = {**_extra_kwargs, "last_n_tokens": 64 if "paged" in attn_name else 1}

    with stagger_region(stagger_update_lazyhandle):
        with torch_sendnn.warmup_mode():
            generate(
                model,
                _warmup_input_ids,
                max_new_tokens=_max_new_tokens,
                do_sample=False,
                use_cache=use_cache,
                extra_kwargs=extra_kwargs,
                **attention_specific_kwargs,
            )
    pt_compile_model_time = time.time() - pt_compile_model_time
    dprint(f"PT compile complete, took {pt_compile_model_time:.3f}s")


def __download_file(url, filename):
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()

        with open(filename, "wb") as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
        print(f"Successfully downloaded {filename}")

    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")


def get_pad_size(prompt_len: int, pad_multiple: int = 64):
    """
    Method to finding nearest prompt length with accepted padding multiple
    i.e.
        prompt length 65 with pad multiple = 64 returns 128
        prompt length 64 with pad multiple = 64 returns 64
    """
    # handle outliers and case where you cannot divide by 0
    if prompt_len <= 0 or pad_multiple == 0:
        if prompt_len < 0:
            warnings.warn(f"{prompt_len=} which should probably be > 0", stacklevel=2)
        return 0
    else:
        return ((prompt_len + pad_multiple - 1) // pad_multiple) * pad_multiple


def _merge_enforce_keep_heterogeneous(
    enforce_list: List[Tuple[str, int]],
    heterogeneous_list: List[Tuple[str, int]],
    batch_size: int,
):
    """
    Method for returning a list that contains both enforced sizes and is heterogeneous

    Args:
        enforce_list: List[Tuple[str, int]], a list of prompt/prompt_len that contains prompt_lens
            that must be enforced
        heterogeneous_list: List[Tuple[str, int]], a list of prompt/prompt_len where all prompt_lens
            are heterogeneous to the extent possible. i.e. if batch size is 3 but only 2 possible
            prompt length exists, this list will contain both prompt lengths with third item sharing
            the same prompt length as one of the previous items.
        batch_size: int, will define the final size of the list.

    Returns:
        List[Tuple[str,int]] that will have all elements from enforce_list
    """
    final_list = enforce_list.copy()
    unique_sizes = {num for _, num in enforce_list}
    for prompt, size in heterogeneous_list:
        if len(final_list) >= batch_size:
            break
        # if the size hasn't been covered by enforce_list, add to list to keep it heterogeneous
        if size not in unique_sizes:
            final_list.append((prompt, size))
            unique_sizes.add(size)
    if len(final_list) > batch_size:
        warnings.warn(
            f"Requested {batch_size=}, which is smaller than the enforced list, will return list larger than requested size",
            stacklevel=2,
        )
    elif len(final_list) < batch_size:
        warnings.warn(
            f"Requested {batch_size=}, is greater than possible combined list. Will return smaller list than batch size",
            stacklevel=2,
        )
    return final_list


def _get_truncation_size(
    dataset_size_and_count: dict[int, int], enforce_sizes: List[int]
):
    """
    Given a list of sizes to enforce and a dictionary of sizes that exists and their count,
    find out which sizes are not possible and create a new truncation list which will grab from
    the next larger size in order to enforce that size.
    If there are no larger sizes, try to take the largest from the dataset.

    Args:
        dataset_size_and_count (Dict[int, int]): List of possible sizes and counts for the dataset
        enforce_sizes (List[int]): List of ints which sizes must be enforced

    Returns:
        List[Tuple[int,int]]: a List of Tuples which have first int as size to truncate to, and second int as to prompt len to grab from
    """
    truncation_list: List[Tuple[int, int]] = []
    sorted_sizes_in_dataset: List[int] = sorted(dataset_size_and_count.keys())
    # sort for consistent results where user mixes order of enforce_sizes
    enforce_sizes = sorted(enforce_sizes)

    for size_to_enforce in enforce_sizes:
        found_idx = bisect.bisect_left(sorted_sizes_in_dataset, size_to_enforce)
        truncation_size = None

        # if valid search found
        if found_idx < len(sorted_sizes_in_dataset):
            while found_idx < len(sorted_sizes_in_dataset):
                # reset the candidate to the new found_idx
                candidate = sorted_sizes_in_dataset[found_idx]
                # Have to check if this prompt length is available with the count
                if dataset_size_and_count[candidate] > 0:
                    # if count is > 0 then decrement the count as it no longer can be used for future prompts
                    dataset_size_and_count[candidate] -= 1
                    truncation_size = candidate
                    break
                # if prompt length is not avaible increment to see if the next larger prompt is available
                found_idx += 1

            if truncation_size is None:
                raise ValueError(
                    f"We've exhausted all possible truncation sizes, please increase max_prompt_len or remove {size_to_enforce=}"
                )
            truncation_list.append((size_to_enforce, truncation_size))
        else:
            # this occurs when size_to_enforce is outside of the max range of dataset
            if sorted_sizes_in_dataset:
                # try to grab the largest size from the end of sorted list if it is available otherwise throw error
                truncation_size = sorted_sizes_in_dataset[-1]
                if dataset_size_and_count[truncation_size] > 0:
                    truncation_list.append((size_to_enforce, truncation_size))
                    dataset_size_and_count[truncation_size] -= 1
                else:
                    raise ValueError(
                        f"{size_to_enforce=} is larger than largest sample and not available."
                    )
    return truncation_list


def _remove_list_from_list(main_list, list_to_remove):
    for item in list_to_remove:
        if item in main_list:
            main_list.remove(item)
    return main_list


# Because we now require encoding the dataset, cache the datasets to make
# second sample request quick
__cached_encoded_datasets = {}


def __sample_requests(
    prompt_list: List[str],
    num_requests: int,
    tokenizer: PreTrainedTokenizerBase,
    prompt_length_min: int = 32,
    prompt_length_max: int = 64,
    seed: Optional[int] = None,
    enforce_heterogeneous: bool = False,
    enforce_sizes: List[int] | None = None,
    truncation: bool = False,
    pad_multiple: int = 64,
    _cached_dataset_key: Optional[str] = None,
):
    """
    Shuffles dataset, tokenizes the prompts and then filters.

    Args:
        prompt_length_min (int): filters out prompts shorter than this value.
        prompt_length_max (int): filters out prompts larger than this value.
        enforce_sizes (List[int]): sample request will grab a prompt with this length if available.
        enforce_heterogeneous (bool): Pads all prompts within batch to nearest multiple of `pad_multiple`.
            However, if enforce_sizes is not empty, it will set enforce_heteogeneous to False.
        pad_multiple (int): Used only when enforce_heterogeneous is True or enforce_sizes is not empty, asserts that prompt_length would be padded to this multiple
        List[Tuple[str, int]]: a filtered dataset
        truncation (bool): If true will truncate to an enforced size if the size does not exist. Only to be used with enforce_sizes, otherwise
        will be ignored
        _cached_dataset_key (optional[str]): The key to the dataset if enabling caching of encoded datasets

    Returns:
        List[Tuple[str, int]]
    """

    assert prompt_length_max >= prompt_length_min, (
        "Please enter valid prompt length max/min values"
    )

    if enforce_sizes is None:
        enforce_sizes = []

    if enforce_heterogeneous and enforce_sizes:
        warnings.warn(
            f"{enforce_heterogeneous=} and {enforce_sizes=}, these two are not designed to be used at the same time. Forcing enforce_heterogeneous to False"
        )
        enforce_heterogeneous = False

    # Based on min/max prompt length, one can back out the number of possible heterogeneous values
    max_heterogeneous_combinations = (prompt_length_max // pad_multiple) - (
        (prompt_length_min - 1) // pad_multiple
    )

    # Filter out sequences that are too long or too short
    dataset: List[Tuple[str, int]] = []
    filtered_dataset: List[Tuple[str, int]] = []
    enforced_dataset: List[Tuple[str, int]] = []

    # To track sizes seen
    seen_sizes: List[int] = []

    sample_size_counter: dict[int, int] = {}
    # first int is the size to truncate to, second int is size of text to grab from
    enforce_sizes_with_truncation: List[Tuple[int, int]] = []

    if truncation and not enforce_sizes:
        warnings.warn(
            f"truncation and enforce_sizes should be used together, whereas {truncation=} and {enforce_sizes=}, hence no truncation will happen",
            stacklevel=2,
        )

    if (
        _cached_dataset_key is not None
        and _cached_dataset_key in __cached_encoded_datasets
    ):
        dataset = __cached_encoded_datasets[_cached_dataset_key]
    else:
        # Loop to check create filtered dataset
        for i in range(len(prompt_list)):
            # Tokenize the prompts and completions.
            prompt = prompt_list[i]
            prompt_token_ids = tokenizer.encode(prompt, return_tensors="pt").squeeze(0)

            prompt_len = len(prompt_token_ids)

            dataset.append((prompt, prompt_len))

        dataset.sort(key=lambda tuple: tuple[1])
        __cached_encoded_datasets[_cached_dataset_key] = dataset

    # only keep values that are required
    dataset = [
        r for r in dataset if r[1] >= prompt_length_min and r[1] <= prompt_length_max
    ]

    pad_size_dict: dict[int, int] = {}
    for _, prompt_len in dataset:
        pad_size_dict.setdefault(prompt_len, get_pad_size(prompt_len, pad_multiple))
        sample_size_counter[pad_size_dict[prompt_len]] = (
            sample_size_counter.get(pad_size_dict[prompt_len], 0) + 1
        )

    if enforce_sizes:
        for size in enforce_sizes:
            # Check that enforced sizes fall within min/max range
            assert prompt_length_min <= size <= prompt_length_max, (
                f"Size {size} in enforced sizes not within {prompt_length_min=}, {prompt_length_max=}"
            )
            assert size % pad_multiple == 0, (
                "Enforce sizes must be a multiple of pad_multiple"
            )
        if len(enforce_sizes) > num_requests:
            raise ValueError(
                f"{num_requests=} which is smaller than {len(enforce_sizes)=}"
            )

        if truncation:
            truncation_size_counter = sample_size_counter.copy()

            # Allocate certain counts to enforce_sizes
            needs_truncation = []
            for size in enforce_sizes:
                if sample_size_counter.get(size, 0) > 0:
                    sample_size_counter[size] -= 1
                else:
                    needs_truncation.append(size)
            enforce_sizes = _remove_list_from_list(enforce_sizes, needs_truncation)

            enforce_sizes_with_truncation = _get_truncation_size(
                truncation_size_counter, needs_truncation
            )

    # Shuffle the dataset.
    if seed is not None:
        random.Random(seed).shuffle(dataset)

    for prompt, prompt_len in dataset:
        if (
            len(filtered_dataset) + len(enforced_dataset) == num_requests
            and not enforce_sizes
        ):
            break

        # NOTE: This section is for enforce heterogeneous, does not work with enforce_sizes
        if (
            enforce_heterogeneous
            and max_heterogeneous_combinations > len(filtered_dataset)
            and len(filtered_dataset) < num_requests
        ):
            current_padded_size = pad_size_dict[prompt_len]

            if current_padded_size not in seen_sizes:
                filtered_dataset.append((prompt, prompt_len))
                seen_sizes.append(current_padded_size)
        # Forcing search for enforce_sizes
        elif enforce_sizes or enforce_sizes_with_truncation:
            current_padded_size = pad_size_dict[prompt_len]
            # if it is in the enforce_size list
            if current_padded_size in enforce_sizes:
                enforce_sizes.remove(current_padded_size)
                enforced_dataset.append((prompt, prompt_len))
            # NOTE: this should not be `elif` despite enforce_sizes and enforce_sizes_with_truncation
            # are mutually exclusive because we allow same prompt to be used in enforce_sizes_with_truncation
            # even if it is taken from enforce_sizes
            truncation_found = None
            if enforce_sizes_with_truncation:
                truncation_found: Tuple[int, int] = next(
                    (
                        tup
                        for tup in enforce_sizes_with_truncation
                        if tup[1] == current_padded_size
                    ),
                    None,
                )
                if truncation_found:
                    truncate_to_size, _ = truncation_found
                    prompt_token_ids = tokenizer.encode(
                        prompt, add_special_tokens=False
                    )
                    # If we don't set clean_up_tokenization_spaces=False, encoding then decoding text might result in different lengths which would break expected results from the sampler
                    truncated_prompt = tokenizer.decode(
                        prompt_token_ids[:truncate_to_size],
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False,
                    )
                    enforced_dataset.append((truncated_prompt, truncate_to_size))
                    enforce_sizes_with_truncation.remove(truncation_found)
            # This condition allows adding prompts to the final dataset as long as there is
            # sufficient space allocated for sizes that need to be enforced.
            if (
                not truncation_found
                and current_padded_size not in enforce_sizes
                and len(filtered_dataset) + len(enforced_dataset)
                < num_requests
                - (len(enforce_sizes) + len(enforce_sizes_with_truncation))
            ):
                filtered_dataset.append((prompt, prompt_len))

        # when not enforcing heterogeneous or when exhausted all possible prompt_lengths
        else:
            filtered_dataset.append((prompt, prompt_len))
    if enforce_sizes:
        warnings.warn(
            f"{enforce_sizes=} so these sizes were not enforced, consider setting truncation=True",
            stacklevel=2,
        )
    if enforce_sizes_with_truncation:
        warnings.warn(
            f"{enforce_sizes_with_truncation=} so not all sizes with truncation enforced",
            stacklevel=2,
        )

    if num_requests > max_heterogeneous_combinations:
        print(
            f"There may be prompt size repeats because {num_requests=} while {max_heterogeneous_combinations=}"
        )
    if enforced_dataset and enforce_heterogeneous:
        filtered_dataset = _merge_enforce_keep_heterogeneous(
            enforced_dataset, filtered_dataset, num_requests
        )
    elif enforced_dataset:
        filtered_dataset = enforced_dataset + filtered_dataset

    if len(filtered_dataset) != num_requests:
        warnings.warn("Returning dataset not equal to number requested", stacklevel=2)

    return filtered_dataset


def sample_rag_factoid_requests(
    dataset_path: str,
    num_requests: int,
    tokenizer: PreTrainedTokenizerBase,
    prompt_length_min: int = 32,
    prompt_length_max: int = 65536,
    seed: Optional[int] = None,
    enforce_heterogeneous: bool = False,
    enforce_sizes: List[int] = [],
    truncation: bool = False,
    pad_multiple: int = 64,
    return_key: bool = False,
) -> List[Tuple[str, int]]:
    if not os.path.exists(dataset_path):
        print("error dataset does not exist")

    dataset = []
    # Load the dataset.
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            dataset.append(line)

    sample_request = __sample_requests(
        dataset,
        num_requests,
        tokenizer,
        prompt_length_min,
        prompt_length_max,
        seed,
        enforce_heterogeneous,
        enforce_sizes,
        truncation,
        pad_multiple,
        _cached_dataset_key=dataset_path,
    )

    if return_key:
        sample_key: str = format_kwargs_to_string(
            dataset="rag_factoid",
            num_requests=num_requests,
            tokenizer=tokenizer.name_or_path.replace("/", "--"),
            prompt_length_min=prompt_length_min,
            prompt_length_max=prompt_length_max,
            seed=seed,
            enforce_heterogeneous=enforce_heterogeneous,
            enforce_sizes=enforce_sizes,
            truncate=truncation,
            pad_multiple=pad_multiple,
        )

        return sample_request, sample_key
    else:
        return sample_request


def sample_sharegpt_requests(
    dataset_path: str,
    num_requests: int,
    tokenizer: PreTrainedTokenizerBase,
    prompt_length_min: int = 32,
    prompt_length_max: int = 64,
    seed: Optional[int] = None,
    enforce_heterogeneous: bool = False,
    enforce_sizes: List[int] | None = None,
    truncation: bool = False,
    pad_multiple: int = 64,
    return_key: bool = False,
) -> List[Tuple[str, int]]:
    if not os.path.exists(dataset_path):
        print("downloading share-gpt dataset as it does not exist")
        is_distributed_initialized = torch.distributed.is_initialized()
        if not is_distributed_initialized or rank < 1:
            __download_file(
                "https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json",
                dataset_path,
            )
        else:
            print("waiting for rank0 to complete download")

        if is_distributed_initialized:
            torch.distributed.barrier()

    if enforce_sizes is None:
        enforce_sizes = []

    # Load the dataset.
    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    # Filter out the conversations with less than 2 turns.
    dataset = [data for data in dataset if len(data["conversations"]) >= 2]
    dataset: List[str] = [data["conversations"][0]["value"] for data in dataset]

    sample_request = __sample_requests(
        dataset,
        num_requests,
        tokenizer,
        prompt_length_min,
        prompt_length_max,
        seed,
        enforce_heterogeneous,
        enforce_sizes,
        truncation,
        pad_multiple,
        _cached_dataset_key=dataset_path,
    )

    if return_key:
        sample_key: str = format_kwargs_to_string(
            dataset="sharegpt",
            num_requests=num_requests,
            tokenizer=tokenizer.name_or_path.replace("/", "--"),
            prompt_length_min=prompt_length_min,
            prompt_length_max=prompt_length_max,
            seed=seed,
            enforce_heterogeneous=enforce_heterogeneous,
            enforce_sizes=enforce_sizes,
            truncate=truncation,
            pad_multiple=pad_multiple,
        )
        return sample_request, sample_key
    else:
        return sample_request


def sample_squad_v2_qa_requests(
    dataset_path: str,
    num_requests: int,
    tokenizer: PreTrainedTokenizerBase,
    prompt_length_min: int = 32,
    prompt_length_max: int = 64,
    seed: Optional[int] = None,
    enforce_heterogeneous: bool = False,
    enforce_sizes: List[int] | None = None,
    truncation: bool = False,
    pad_multiple: int = 64,
    return_key: bool = False,
) -> List[Tuple[str, int]]:
    from datasets import load_dataset

    if enforce_sizes is None:
        enforce_sizes = []

    if os.path.exists(dataset_path):
        ds = load_dataset(dataset_path)["train"]
    else:
        ds = load_dataset("rajpurkar/squad_v2", cache_dir=dataset_path)["train"]

    ds = [f"{data['context']}\n{data['question']}" for data in ds]

    sample_request = __sample_requests(
        ds,
        num_requests,
        tokenizer,
        prompt_length_min,
        prompt_length_max,
        seed,
        enforce_heterogeneous,
        enforce_sizes,
        truncation,
        pad_multiple,
    )

    if return_key:
        sample_key: str = format_kwargs_to_string(
            dataset="squad_v2",
            num_requests=num_requests,
            tokenizer=tokenizer.name_or_path.replace("/", "--"),
            prompt_length_min=prompt_length_min,
            prompt_length_max=prompt_length_max,
            seed=seed,
            enforce_heterogeneous=enforce_heterogeneous,
            enforce_sizes=enforce_sizes,
            truncate=truncation,
            pad_multiple=pad_multiple,
        )
        return sample_request, sample_key
    else:
        return sample_request


def prepare_inputs(
    batch_size, seq_length, tokenizer, ds_path, seed=0, ds_type="sharegpt"
):
    """
    Prepare input IDs and padding kwargs for a batch of questions.

    Args:
        batch_size (int): The number of questions in the batch.
        seq_length (int): The maximum length of the input sequence.
        tokenizer (Tokenizer): A tokenizer object to tokenize the questions.
        ds_path (str): The path to the dataset file.
        seed (int, optional): The random seed for reproducibility. Defaults to 0.
        ds_type (str, optional): The type of dataset to use. Can be "sharegpt" or any other supported dataset type. Defaults to "sharegpt".

    Returns:
        tuple: A tuple containing the input IDs and padding kwargs.
    """
    if "sharegpt" not in ds_type:
        prompts_and_sizes = sample_squad_v2_qa_requests(
            ds_path,
            batch_size,
            tokenizer,
            int(seq_length / 2),
            seq_length,
            seed,
        )
    else:
        prompts_and_sizes = sample_sharegpt_requests(
            ds_path,
            batch_size,
            tokenizer,
            int(seq_length / 2),
            seq_length,
            seed,
        )
    prompt_list = []
    for prompt, _ in prompts_and_sizes:
        prompt_list.append(tokenizer.encode(prompt, return_tensors="pt").squeeze(0))

    input_ids, padding_kwargs = pad_input_ids(prompt_list, min_pad_length=seq_length)
    return input_ids, padding_kwargs
