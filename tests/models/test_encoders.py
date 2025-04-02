from fms.testing.comparison import ModelSignatureParams, compare_model_signatures, get_signature
from fms.utils import tokenizers
import pytest
from fms.models import get_model
from fms.utils.generation import pad_input_ids
import itertools
import torch
from aiu_fms_testing_utils.utils import ids_for_prompt, sample_squad_v2_qa_requests
from aiu_fms_testing_utils.utils.aiu_setup import dprint
import os

ORIGINAL_HF_HOME = os.environ.get("HF_HOME", None)

# Add models to test here
ROBERTA_SQUAD_V2 = "deepset/roberta-base-squad2"

SQUAD_V2_DATASET_PATH = os.environ.get("SQUAD_V2_DATASET_PATH", os.path.expanduser("~/squad_v2"))
common_model_paths = os.environ.get("FMS_TEST_SHAPES_COMMON_MODEL_PATHS", [ROBERTA_SQUAD_V2])
common_batch_sizes = os.environ.get("FMS_TEST_SHAPES_COMMON_BATCH_SIZES", [1, 2, 4, 8])
common_seq_lengths = os.environ.get("FMS_TEST_SHAPES_COMMON_SEQ_LENGTHS", [64, 512])

# pass custom model path list for eg: EXPORT FMS_TESTING_COMMON_MODEL_PATHS="/tmp/models/roberta,/tmp/models/roberta-base-squad2"
if isinstance(common_model_paths, str):
    common_model_paths = common_model_paths.split(",")

# pass custom common batch sizes as a comma separated str of ints
if isinstance(common_batch_sizes, str):
    common_batch_sizes = [int(bs) for bs in common_batch_sizes.split(",")]

# pass custom common seq lengths as a comma separated str of ints
if isinstance(common_seq_lengths, str):
    common_seq_lengths = [int(sl) for sl in common_seq_lengths.split(",")]

common_shapes = list(itertools.product(common_model_paths, common_batch_sizes, common_seq_lengths))


def __prepare_inputs(batch_size, seq_length, tokenizer, seed=0):
    prompts_and_sizes = sample_squad_v2_qa_requests(SQUAD_V2_DATASET_PATH, batch_size, tokenizer, int(seq_length / 2), seq_length, seed)
    prompt_list = []
    for prompt, _ in prompts_and_sizes:
        prompt_list.append(ids_for_prompt(prompt, tokenizer))

    input_ids, padding_kwargs = pad_input_ids(prompt_list, min_pad_length=seq_length, is_causal_mask=False)
    return input_ids, padding_kwargs

@pytest.fixture(autouse=True)
def reset_compiler():
    yield # run the test
    torch.compiler.reset()
    torch._dynamo.reset()
    os.environ.pop('COMPILATION_MODE', None)
    if ORIGINAL_HF_HOME is None:
        os.environ.pop('HF_HOME', None)
    else:
        os.environ['HF_HOME'] = ORIGINAL_HF_HOME

encoder_paths = ["deepset/roberta-base-squad2"]
common_encoder_shapes = list(itertools.product(encoder_paths, common_batch_sizes, common_seq_lengths))

@pytest.mark.parametrize("model_path,batch_size,seq_length", common_encoder_shapes)
def test_common_shapes(model_path, batch_size, seq_length):
    os.environ["COMPILATION_MODE"] = "offline"

    if "HF_HOME" not in os.environ:
        os.environ["HF_HOME"] = "/tmp/models/hf_cache"
    
    dprint(f"testing model={model_path}, batch_size={batch_size}, seq_length={seq_length}")

    tokenizer = tokenizers.get_tokenizer(model_path)
    
    if os.path.exists(model_path):
        model_path_kwargs = {"model_path": model_path}
    else:
        model_path_kwargs = {"variant": model_path}

    # prepare the AIU model
    model = get_model(
        architecture="hf_pretrained",
        device_type="cpu",
        fused_weights=False,
        **model_path_kwargs
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
        **model_path_kwargs
    )

    # prepare input_ids
    input_ids, padding_kwargs = __prepare_inputs(batch_size, seq_length, tokenizer)

    # warmup model
    logits_getter_fn = lambda x: x if isinstance(x, torch.Tensor) else torch.cat(list(x), dim=-1)
    aiu_msp = ModelSignatureParams(model, ["x"], logits_getter_fn=logits_getter_fn, inp=input_ids, other_params=padding_kwargs)
    get_signature(aiu_msp.model, aiu_msp.params, aiu_msp.inp, aiu_msp.other_params, aiu_msp.logits_getter_fn)

    cpu_msp = ModelSignatureParams(validation_model, ["x"], logits_getter_fn=logits_getter_fn, inp=input_ids, other_params=padding_kwargs)
    # FIXME: Compute GPU atol/rtol
    compare_model_signatures(cpu_msp, aiu_msp, atol=0.1, rtol=.05)

def test_warmup_multiple_shapes():
    os.environ["COMPILATION_MODE"] = "offline"

    if "HF_HOME" not in os.environ:
        os.environ["HF_HOME"] = "/tmp/models/hf_cache"
    
    model_path = "deepset/roberta-base-squad2"
    tokenizer = tokenizers.get_tokenizer(model_path)
    
    if os.path.exists(model_path):
        model_path_kwargs = {"model_path": model_path}
    else:
        model_path_kwargs = {"variant": model_path}

    
    shapes = [
        (1, 64),
        (2, 64),
        (1, 128),
    ]

    # prepare the AIU model
    model = get_model(
        architecture="hf_pretrained",
        device_type="cpu",
        fused_weights=False,
        **model_path_kwargs
    )

    model.eval()
    torch.set_grad_enabled(False)
    model.compile(backend="sendnn")

    # prepare the cpu model
    reference_model = get_model(
        architecture="hf_pretrained",
        device_type="cpu",
        data_type=torch.float32,
        fused_weights=False,
        **model_path_kwargs
    )

    # encoders should be using static shapes
    with torch._dynamo.config.patch(
        assume_static_by_default=True,
        dynamic_shapes=False,
        automatic_dynamic_shapes=False,
        cache_size_limit=1000,
    ):

        for batch_size, seq_length in shapes:

            # prepare input_ids
            input_ids, padding_kwargs = __prepare_inputs(batch_size, seq_length, tokenizer)

            # warmup model
            logits_getter_fn = lambda x: x if isinstance(x, torch.Tensor) else torch.cat(list(x), dim=-1)
            aiu_msp = ModelSignatureParams(model, ["x"], logits_getter_fn=logits_getter_fn, inp=input_ids, other_params=padding_kwargs)
            get_signature(aiu_msp.model, aiu_msp.params, aiu_msp.inp, aiu_msp.other_params, aiu_msp.logits_getter_fn)
        
        for _ in range(3):
            shapes.reverse()

            for batch_size, seq_length in shapes:
                logits_getter_fn = lambda x: x if isinstance(x, torch.Tensor) else torch.cat(list(x), dim=-1)
                aiu_msp = ModelSignatureParams(model, ["x"], logits_getter_fn=logits_getter_fn, inp=input_ids, other_params=padding_kwargs)
                get_signature(aiu_msp.model, aiu_msp.params, aiu_msp.inp, aiu_msp.other_params, aiu_msp.logits_getter_fn)

                cpu_msp = ModelSignatureParams(reference_model, ["x"], logits_getter_fn=logits_getter_fn, inp=input_ids, other_params=padding_kwargs)
                compare_model_signatures(cpu_msp, aiu_msp, atol=0.1, rtol=.05)


    