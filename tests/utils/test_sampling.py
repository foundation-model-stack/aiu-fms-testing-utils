import os
from itertools import product
from typing import List

import pytest
from transformers import AutoTokenizer

from aiu_fms_testing_utils.utils import (_merge_enforce_keep_heterogeneous,
                                         get_pad_size,
                                         sample_sharegpt_requests)

BATCH_SIZES = [0, 1, 2, 3, 4, 8, 16]
ENFORCE_HETEROGENEOUS = [True, False]
ENFORCE_SIZES = [[], [64, 256], [64, 128, 2048, 4096]]
SEED = [0, 1, 15, 256]
PAD_SIZES = [0, 64, 128, 256]

prompt_max_length = 4096
prompt_min_length = 64
enforce_heterogeneous = True


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
    product(BATCH_SIZES, ENFORCE_HETEROGENEOUS, ENFORCE_SIZES, SEED)
)


@pytest.mark.parametrize(
    "batch_size, enforce_heterogeneous, enforce_sizes, seed", ENFORCE_TEST_COMBO
)
def test_enforce_heterogeneous_and_size(
    batch_size, enforce_heterogeneous, enforce_sizes, seed
):
    prompt_max_length = 4096
    prompt_min_length = 64
    dataset = os.environ.get(
        "SHARE_GPT_DATASET_PATH", os.path.expanduser("~/share_gpt.json")
    )
    enforce_size_copy = enforce_sizes.copy()
    tokenizer = AutoTokenizer.from_pretrained("ibm-granite/granite-3.3-8b-instruct")
    if batch_size < len(enforce_size_copy):
        with pytest.raises(
            ValueError, match="num request is smaller than enforce_sizes"
        ):
            expected_error(batch_size, enforce_size_copy)
    else:
        prompts_and_sizes = sample_sharegpt_requests(
            dataset,
            batch_size,
            tokenizer,
            prompt_min_length,
            prompt_max_length,
            seed,
            enforce_heterogeneous,
            enforce_sizes,
        )
        if enforce_size_copy and enforce_heterogeneous:
            # check enforce size
            for size in enforce_size_copy:
                assert size in [get_pad_size(item[1]) for item in prompts_and_sizes]
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
            for size in enforce_size_copy:
                assert size in [get_pad_size(item[1]) for item in prompts_and_sizes]
        # verify the right size is returned
        assert len(prompts_and_sizes) == batch_size
