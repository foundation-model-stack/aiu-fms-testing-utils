import os

import torch
from fms.models import get_model
from fms.utils.generation import generate, pad_input_ids
from fms.utils.tokenizers import get_tokenizer

from aiu_fms_testing_utils.utils.paged import generate as paged_generate


def test_paged_equivalence():
    torch.manual_seed(0)
    with torch.no_grad():
        _model_mock = get_model("gpt_bigcode", "micro")
        _model_mock.reset_parameters()
        _model_mock.eval()
        tokenizer = get_tokenizer("char_tokenizer")
        first = torch.tensor(
            tokenizer.convert_tokens_to_ids(tokenizer.tokenize("ABCDE")),
            dtype=torch.long,
        )
        second = torch.tensor(
            tokenizer.convert_tokens_to_ids(tokenizer.tokenize("CDEFGHIJKL")),
            dtype=torch.long,
        )

        # use_cache=True
        ids, padding_kwargs = pad_input_ids([first, second], min_pad_length=64)
        result = generate(
            _model_mock,
            ids,
            max_seq_len=ids.shape[1] + 5,
            max_new_tokens=5,
            do_sample=False,
            use_cache=True,
            extra_kwargs=padding_kwargs,
        )
        os.environ["VLLM_DT_MAX_BATCH_SIZE"] = "2"
        os.environ["VLLM_DT_MAX_CONTEXT_LEN"] = "256"

        result_paged = paged_generate(
            _model_mock,
            ids,
            max_new_tokens=5,
            do_sample=False,
            use_cache=True,
            extra_kwargs={"attn_name": "spyre_paged_attn", **padding_kwargs},
        )
        torch.testing.assert_close(result, result_paged)
