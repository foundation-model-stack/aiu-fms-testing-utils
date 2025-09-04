from aiu_fms_testing_utils.utils import (
    get_pad_size,
    _merge_enforce_keep_heterogeneous,
    _get_truncation_size,
    __sample_requests,
)
from typing import List
from transformers import AutoTokenizer
import pytest
from itertools import product
import os
import json

BATCH_SIZES = [0, 1, 2, 3, 4, 8]
ENFORCE_HETEROGENEOUS = [True, False]
TRUNCATION = [True, False]
ENFORCE_TRUNCATION_SIZE = [
    [],
    [64],
    [128],
    [64, 64],
    [64, 128],
    [64, 128, 256],
    [64, 64, 64, 64],
]
AVAILABLE_SIZES = [
    {},
    {64: 1, 128: 1},
    {128: 1, 256: 1},
    {64: 1, 128: 1, 256: 1},
    {64: 1, 2048: 2},
]
ENFORCE_SIZES_SHAREGPT = [[], [6208, 6272], [6208, 6400, 7168, 8192]]
SHAREGPT_SUBSAMPlE_SIZE_AND_COUNT = {
    6144: 2,
    6208: 1,
    6272: 3,
    6400: 2,
    6464: 1,
    6528: 2,
    6592: 1,
    6720: 1,
    6784: 1,
    6848: 3,
    6976: 2,
    7104: 1,
    7232: 1,
    7296: 2,
    7360: 2,
    7488: 1,
    7872: 3,
    8128: 1,
}
SEED = [0, 256]
PAD_SIZES = [0, 64, 128, 256]
PROMPT_MAX_LENGTH = 8192
PROMPT_MIN_LENGTH = 6144
TOKENIZER = AutoTokenizer.from_pretrained("ibm-granite/granite-3.3-8b-instruct")


def _prepare_sub_sharegpt_dataset(prompt_length_min, prompt_length_max, tokenizer):
    dataset_path = os.environ.get(
        "SHARE_GPT_DATASET_PATH", os.path.expanduser("~/share_gpt.json")
    )
    # Load the dataset.
    with open(dataset_path, encoding="utf-8") as f:
        prompt_list = json.load(f)
    # Filter out the conversations with less than 2 turns.
    prompt_list = [data for data in prompt_list if len(data["conversations"]) >= 2]
    prompt_list: List[str] = [data["conversations"][0]["value"] for data in prompt_list]

    dataset: List[str] = []

    # Loop to check create filtered dataset
    batch_encoding = tokenizer(prompt_list, return_length=True)
    for prompt, prompt_len in zip(prompt_list, batch_encoding.length):
        if prompt_len < prompt_length_min or prompt_len > prompt_length_max:
            # Prune too short or too long sequences.
            continue

        dataset.append(prompt)
    return dataset


@pytest.fixture(scope="session")
def sub_sharegpt_dataset():
    return _prepare_sub_sharegpt_dataset(
        PROMPT_MIN_LENGTH, PROMPT_MAX_LENGTH, TOKENIZER
    )


def expected_error(num_request: int, enforce_sizes: List[int]):
    if num_request < len(enforce_sizes):
        raise ValueError("num request is smaller than enforce_sizes")
    return "OK"


@pytest.mark.parametrize("batch_size", BATCH_SIZES)
def test_merge_enforce_keep_heterogeneous(batch_size):
    """
    testing that all items in keep_list are kept while returning correct batch size by populating
    final_list from flex_list and keeping everything heterogeneous
    """
    num_keep = 0
    num_flex = 0
    keep_list = [("keep", 0), ("keep", 2), ("keep", 3)]
    flexible_list = [("flex", 2), ("flex", 3), ("flex", 4), ("flex", 5), ("flex", 6)]
    final_list = _merge_enforce_keep_heterogeneous(keep_list, flexible_list, batch_size)
    for text, _ in final_list:
        if text == "keep":
            num_keep += 1
        else:
            num_flex += 1
    assert num_keep == len(keep_list)
    if batch_size <= len(keep_list):
        assert num_flex == 0
    else:
        len_unique_num = len(set([item[1] for item in keep_list + flexible_list]))
        assert num_flex == min(batch_size - num_keep, len_unique_num - num_keep)


@pytest.mark.parametrize("expected_pad_size", PAD_SIZES)
def test_get_pad_size(expected_pad_size):
    # check default 64
    assert get_pad_size(63) == 64

    assert get_pad_size(0, expected_pad_size) == 0
    assert get_pad_size(expected_pad_size - 1, expected_pad_size) == expected_pad_size
    assert get_pad_size(expected_pad_size, expected_pad_size) == expected_pad_size
    assert (
        get_pad_size(expected_pad_size + 1, expected_pad_size) == 2 * expected_pad_size
    )
    assert get_pad_size(-1, expected_pad_size) == 0


ENFORCE_TEST_COMBO = list(
    product(
        BATCH_SIZES, ENFORCE_HETEROGENEOUS, ENFORCE_SIZES_SHAREGPT, SEED, TRUNCATION
    )
)


@pytest.mark.parametrize(
    "batch_size, enforce_heterogeneous, enforce_sizes, seed, truncation",
    ENFORCE_TEST_COMBO,
)
def test_enforce_heterogeneous_and_size(
    batch_size,
    enforce_heterogeneous,
    enforce_sizes,
    seed,
    truncation,
    sub_sharegpt_dataset,
):
    enforce_size_copy = enforce_sizes.copy()

    if batch_size < len(enforce_size_copy):
        with pytest.raises(
            ValueError, match="num request is smaller than enforce_sizes"
        ):
            expected_error(batch_size, enforce_size_copy)
    else:
        prompts_and_sizes = __sample_requests(
            sub_sharegpt_dataset,
            batch_size,
            TOKENIZER,
            PROMPT_MIN_LENGTH,
            PROMPT_MAX_LENGTH,
            seed,
            enforce_heterogeneous,
            enforce_sizes,
            truncation,
        )
        enforceable_without_truncation = True
        if enforce_size_copy and enforce_heterogeneous:
            # check enforce size
            for size_to_enforce in enforce_size_copy:
                if size_to_enforce in SHAREGPT_SUBSAMPlE_SIZE_AND_COUNT.keys():
                    assert size_to_enforce in [
                        get_pad_size(size) for _, size in prompts_and_sizes
                    ]
                else:
                    enforceable_without_truncation = False
            # check heterogeneous
            assert len(prompts_and_sizes) == len(
                set(item[1] for item in prompts_and_sizes)
            )
        elif not enforce_size_copy and enforce_heterogeneous:
            # check heterogeneous
            assert len(prompts_and_sizes) == len(
                set(item[1] for item in prompts_and_sizes)
            )
        elif enforce_size_copy and not enforce_heterogeneous:
            # check enforce size
            for size_to_enforce in enforce_size_copy:
                if size_to_enforce in SHAREGPT_SUBSAMPlE_SIZE_AND_COUNT.keys():
                    assert size_to_enforce in [
                        get_pad_size(size) for _, size in prompts_and_sizes
                    ]
                else:
                    enforceable_without_truncation = False

        # verify the right size is returned
        if enforceable_without_truncation:
            assert len(prompts_and_sizes) == batch_size
        else:
            # enforce_size logic tries to enforce size first before populating rest of batch size.
            # Hence, if it gets stuck trying to find an enforceable size, it will end up with smaller batch.
            assert len(prompts_and_sizes) <= batch_size


ENFORCE_TRUNCATION_COMBO = list(product(ENFORCE_TRUNCATION_SIZE, AVAILABLE_SIZES))


@pytest.mark.parametrize(
    "enforce_truncation_size, available_sizes", ENFORCE_TRUNCATION_COMBO
)
def test_get_truncation(enforce_truncation_size, available_sizes):
    start_available_sizes = available_sizes.copy()
    end_available_sizes = available_sizes.copy()
    try:
        truncation_list = _get_truncation_size(
            end_available_sizes, enforce_truncation_size
        )
        if not enforce_truncation_size:
            assert not truncation_list
        expected_num_truncate = 0
        for size in enforce_truncation_size:
            if size not in end_available_sizes.keys() and end_available_sizes.keys():
                expected_num_truncate += 1
        # even if the size is in available sizes, you may still end up adding to truncation list
        assert len(truncation_list) >= expected_num_truncate
        if end_available_sizes.keys() == 0:
            assert expected_num_truncate == 0
        assert sum(start_available_sizes.values()) - sum(
            end_available_sizes.values()
        ) == len(truncation_list)

        # check count never goes below 0
        for count in available_sizes.values():
            assert count >= 0
    except ValueError as e:
        assert "size_to_enforce" in f"{e}"
    except Exception as e:
        pytest.fail(f"Unexpeced exception: {e}")
