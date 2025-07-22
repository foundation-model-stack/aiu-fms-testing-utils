import tempfile
import pytest
from aiu_fms_testing_utils.testing.validation import (
    LogitsExtractorHook,
    extract_validation_information,
    load_validation_information,
)
from fms.models import get_model
from fms.utils.generation import pad_input_ids
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
