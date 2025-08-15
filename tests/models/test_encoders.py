from fms.testing.comparison import (
    ModelSignatureParams,
    get_signature,
)
import pytest
from fms.models import get_model
from fms.utils.generation import pad_input_ids
import itertools
import torch
from aiu_fms_testing_utils.utils import sample_squad_v2_qa_requests
from aiu_fms_testing_utils.utils.aiu_setup import dprint
import os
import numpy as np
from transformers import AutoTokenizer

# Add models to test here
ROBERTA_SQUAD_V2 = "deepset/roberta-base-squad2"

SQUAD_V2_DATASET_PATH = os.environ.get(
    "SQUAD_V2_DATASET_PATH", os.path.expanduser("~/squad_v2")
)
common_model_paths = os.environ.get(
    "FMS_TEST_SHAPES_COMMON_MODEL_PATHS", [ROBERTA_SQUAD_V2]
)
common_batch_sizes = os.environ.get("FMS_TEST_SHAPES_COMMON_BATCH_SIZES", [1, 2, 4, 8])
common_seq_lengths = os.environ.get("FMS_TEST_SHAPES_COMMON_SEQ_LENGTHS", [64, 512])
validation_diff_threshold = os.environ.get(
    "FMS_TEST_SHAPES_VALIDATION_DIFF_THRESHOLD", 0.01
)

# pass custom model path list for eg: EXPORT FMS_TESTING_COMMON_MODEL_PATHS="/tmp/models/roberta,/tmp/models/roberta-base-squad2"
if isinstance(common_model_paths, str):
    common_model_paths = common_model_paths.split(",")

# pass custom common batch sizes as a comma separated str of ints
if isinstance(common_batch_sizes, str):
    common_batch_sizes = [int(bs) for bs in common_batch_sizes.split(",")]

# pass custom common seq lengths as a comma separated str of ints
if isinstance(common_seq_lengths, str):
    common_seq_lengths = [int(sl) for sl in common_seq_lengths.split(",")]

# FIXME: compare with GPU diffs for default value
# pass custom validation diff threshold (if average of absolute mean diff of all samples < validation_diff_threshold, pass test)
if isinstance(validation_diff_threshold, str):
    validation_diff_threshold = float(validation_diff_threshold)

common_shapes = list(
    itertools.product(common_model_paths, common_batch_sizes, common_seq_lengths)
)


def __prepare_inputs(batch_size, seq_length, tokenizer, seed=0):
    prompts_and_sizes = sample_squad_v2_qa_requests(
        SQUAD_V2_DATASET_PATH,
        batch_size,
        tokenizer,
        int(seq_length / 2),
        seq_length,
        seed,
    )
    prompt_list = []
    for prompt, _ in prompts_and_sizes:
        prompt_list.append(tokenizer.encode(prompt, return_tensors="pt").squeeze(0))

    input_ids, padding_kwargs = pad_input_ids(
        prompt_list, min_pad_length=seq_length, is_causal_mask=False
    )
    return input_ids, padding_kwargs


def __generate_diffs(model_params_1, model_params_2):
    model_params_1.model.eval()
    model_params_2.model.eval()
    signature = get_signature(
        model_params_1.model,
        params=model_params_1.params,
        optional_params=model_params_1.other_params,
        logits_getter_fn=model_params_1.logits_getter_fn,
        inp=model_params_1.inp,
        device=model_params_1.inp.device,
    )
    signature2 = get_signature(
        model_params_2.model,
        params=model_params_2.params,
        optional_params=model_params_2.other_params,
        logits_getter_fn=model_params_2.logits_getter_fn,
        inp=model_params_2.inp,
        device=model_params_2.inp.device,
    )

    signature = np.array(signature)
    signature2 = np.array(signature2)

    return np.mean(np.abs(signature2 - signature))


@pytest.fixture(autouse=True)
def reset_compiler():
    yield  # run the test
    torch.compiler.reset()
    torch._dynamo.reset()
    os.environ.pop("COMPILATION_MODE", None)


@pytest.mark.parametrize("model_path,batch_size,seq_length", common_shapes)
def test_common_shapes(model_path, batch_size, seq_length):
    os.environ["COMPILATION_MODE"] = "offline"

    dprint(
        f"testing model={model_path}, batch_size={batch_size}, seq_length={seq_length}"
    )

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    if os.path.exists(model_path):
        model_path_kwargs = {"model_path": model_path}
    else:
        model_path_kwargs = {"variant": model_path}

    # prepare the AIU model
    model = get_model(
        architecture="hf_pretrained",
        device_type="cpu",
        fused_weights=False,
        **model_path_kwargs,
    )

    model.eval()
    torch.set_grad_enabled(False)
    model.compile(backend="sendnn")

    # prepare the cpu model
    validation_model = get_model(
        architecture="hf_pretrained",
        device_type="cpu",
        data_type=torch.float32,
        fused_weights=False,
        **model_path_kwargs,
    )

    # prepare input_ids
    input_ids, padding_kwargs = __prepare_inputs(batch_size, seq_length, tokenizer)

    # warmup model
    logits_getter_fn = (  # noqa: E731
        lambda x: x if isinstance(x, torch.Tensor) else torch.cat(list(x), dim=-1)
    )
    aiu_msp = ModelSignatureParams(
        model,
        ["x"],
        logits_getter_fn=logits_getter_fn,
        inp=input_ids,
        other_params=padding_kwargs,
    )
    get_signature(
        aiu_msp.model,
        aiu_msp.params,
        aiu_msp.inp,
        aiu_msp.other_params,
        aiu_msp.logits_getter_fn,
    )

    # get the average diff over multiple samples
    diffs = []
    for i in range(20):
        # prepare input_ids
        input_ids, padding_kwargs = __prepare_inputs(
            batch_size, seq_length, tokenizer, seed=i
        )

        aiu_msp = ModelSignatureParams(
            model,
            ["x"],
            logits_getter_fn=logits_getter_fn,
            inp=input_ids,
            other_params=padding_kwargs,
        )
        cpu_msp = ModelSignatureParams(
            validation_model,
            ["x"],
            logits_getter_fn=logits_getter_fn,
            inp=input_ids,
            other_params=padding_kwargs,
        )
        diffs.append(__generate_diffs(aiu_msp, cpu_msp))

    abs_mean_diff = sum(diffs) / len(diffs)
    print(f"absolute mean diff: {abs_mean_diff}")

    assert abs_mean_diff < validation_diff_threshold
