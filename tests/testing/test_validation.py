import re
import tempfile
import pytest
from aiu_fms_testing_utils.testing.validation import (
    LogitsExtractorHook,
    extract_validation_information,
    load_validation_information,
    get_validation_info_path,
    find_validation_info_path,
    __decrement_version,
)
from aiu_fms_testing_utils._version import version_tuple
from fms.models import get_model
from fms.utils.generation import pad_input_ids
from pathlib import Path
import torch


@pytest.mark.parametrize(
    "validation_type,post_iteration_hook",
    [("logits", LogitsExtractorHook()), ("tokens", None)],
)
def test_validation_info_round_trip(validation_type, post_iteration_hook):
    # prepare a small cpu model
    model = get_model(
        "llama",
        "micro",
        device_type="cpu",
    )
    model.reset_parameters()

    seq_length = 64
    batch_size = 8
    max_new_tokens = 128

    # prepare input_ids
    prompt_list = []
    for i in range(batch_size):
        prompt_list.append(
            torch.randint(
                0, model.config.src_vocab_size, (seq_length - 2 * i,), dtype=torch.long
            )
        )

    input_ids, padding_kwargs = pad_input_ids(prompt_list, min_pad_length=seq_length)

    # generate cpu validation info
    generated_validation_info = extract_validation_information(
        model,
        input_ids,
        max_new_tokens,
        post_iteration_hook,
        attn_algorithm="math",
        **padding_kwargs,
    )

    with tempfile.TemporaryDirectory() as workdir:
        output_path = f"{workdir}/validation_info"
        generated_validation_info.save(output_path)

        loaded_validation_info = load_validation_information(
            output_path, validation_type, batch_size
        )

        assert len(generated_validation_info) == len(loaded_validation_info)

        for gen_vi, loaded_vi in zip(generated_validation_info, loaded_validation_info):
            gen_vi_no_none = {k: v for k, v in gen_vi.items() if v is not None}
            loaded_vi_no_none = {k: v for k, v in loaded_vi.items() if v is not None}
            assert gen_vi_no_none.keys() == loaded_vi_no_none.keys()
            for k in gen_vi_no_none.keys():
                torch.testing.assert_close(gen_vi_no_none[k], loaded_vi_no_none[k])


def test_get_validation_info_path(tmp_path):
    assert (
        get_validation_info_path(
            tmp_path, "ibm-granite/granite-3.3-8b-instruct", 4, 64, 128, 0, "sdpa"
        )
        == f"{tmp_path}/ibm-granite--granite-3.3-8b-instruct_max-new-tokens-128_batch-size-4_seq-length-64_dtype-fp16_attn-type-sdpa.{'.'.join([str(_) for _ in version_tuple[:3]])}.cpu_validation_info.0.out"
    )
    assert (
        get_validation_info_path(
            tmp_path,
            "ibm-granite/granite-3.3-8b-instruct",
            4,
            64,
            128,
            0,
            "sdpa",
            aftu_version=(1, 2, 3),
        )
        == f"{tmp_path}/ibm-granite--granite-3.3-8b-instruct_max-new-tokens-128_batch-size-4_seq-length-64_dtype-fp16_attn-type-sdpa.1.2.3.cpu_validation_info.0.out"
    )


@pytest.mark.parametrize(
    "current_version,save_version,expected_version,version_allow_decrement",
    [
        (
            None,
            version_tuple[:3],
            version_tuple[:3],
            True,
        ),  # saved version is the same version as current - find
        (
            (1, 1, 1),
            (1, 1, 2),
            None,
            True,
        ),  # current version is less than any saved version -- dont find - micro
        (
            (0, 0, 3),
            (0, 1, 2),
            None,
            True,
        ),  # current version is less than any saved version -- dont find - minor
        (
            (0, 2, 3),
            (1, 1, 2),
            None,
            True,
        ),  # current version is less than any saved version -- dont find - major
        (
            (1, 1, 2),
            (1, 1, 1),
            (1, 1, 1),
            True,
        ),  # current version is greater than saved version -- find saved version - micro
        (
            (1, 1, 2),
            (1, 1, 0),
            (1, 1, 0),
            True,
        ),  # current version is greater than saved version -- find saved version - minor
        (
            (1, 1, 2),
            (1, 0, 0),
            (1, 0, 0),
            True,
        ),  # current version is greater than saved version -- find saved version - major
        (
            (1, 1, 2),
            (1, 1, 1),
            None,
            False,
        ),  # current version is greater than saved version -- dont find - micro - no decrement
        (
            (1, 1, 2),
            (1, 1, 0),
            None,
            False,
        ),  # current version is greater than saved version -- dont find - minor - no decrement
        (
            (1, 1, 2),
            (1, 0, 0),
            None,
            False,
        ),  # current version is greater than saved version -- dont find - major - no decrement
    ],
)
def test_find_validation_info_path(
    current_version, save_version, expected_version, version_allow_decrement, tmp_path
):
    # create a large version path to make sure we never choose it
    large_version_path = Path(
        get_validation_info_path(
            tmp_path,
            "ibm-granite/granite-3.3-8b-instruct",
            4,
            64,
            128,
            0,
            "sdpa",
            (10, 10, 10),
        )
    )
    large_version_path.write_text("test")
    assert large_version_path.exists()

    save_path = Path(
        get_validation_info_path(
            tmp_path,
            "ibm-granite/granite-3.3-8b-instruct",
            4,
            64,
            128,
            0,
            "sdpa",
            save_version,
        )
    )
    save_path.write_text("test")
    assert save_path.exists()

    found_path = find_validation_info_path(
        tmp_path,
        "ibm-granite/granite-3.3-8b-instruct",
        4,
        64,
        128,
        0,
        "sdpa",
        current_version,
        version_allow_decrement=version_allow_decrement,
    )

    if expected_version is None:
        assert found_path is None
    else:
        match = re.search(r"(\d+)\.(\d+)\.(\d+)", found_path)
        found_version = (int(match.group(1)), int(match.group(2)), int(match.group(3)))
        assert found_version == expected_version


@pytest.mark.parametrize(
    "max_minor,max_patch,current_version",
    [
        [9, 9, (2, 2, 1)],
        [5, 4, (2, 2, 1)],
        [9, 9, (1, 2, 3)],
        [6, 3, (1, 2, 3)],
        [9, 9, (0, 3, 0)],
    ],
)
def test_decrement_version(max_minor, max_patch, current_version):
    major, minor, patch = current_version
    counter = 0
    while current_version is not None:
        current_version = __decrement_version(
            current_version, max_minor=max_minor, max_patch=max_patch
        )
        counter += 1
    assert (
        counter
        == major * (max_minor + 1) * (max_patch + 1)
        + minor * (max_patch + 1)
        + patch
        + 1
    )
