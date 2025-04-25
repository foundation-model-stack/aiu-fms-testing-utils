from fms.models import get_model
from fms.utils.generation import pad_input_ids
from fms.utils.tokenizers import get_tokenizer
import pytest
from aiu_fms_testing_utils.utils import sample_squad_v2_qa_requests
import torch
from aiu_fms_testing_utils.utils import (
    ids_for_prompt,
)

from fms.testing._internal.model_test_suite import (
    ModelConsistencyTestSuite,
    ModelFixtureMixin,
)
import os

os.environ["COMPILATION_MODE"] = "offline"

if "HF_HOME" not in os.environ:
    os.environ["HF_HOME"] = "/tmp/models/hf_cache"

model_dir = os.environ.get("FMS_TESTING_MODEL_DIR", "/tmp/models")
LLAMA_3p1_8B_INSTRUCT = "meta-llama/Llama-3.1-8B-Instruct"
GRANITE_3p2_8B_INSTRUCT = "ibm-granite/granite-3.2-8b-instruct"
GRANITE_20B_CODE_INSTRUCT_8K = "ibm-granite/granite-20b-code-instruct-8k"
LLAMA_3p1_70B_INSTRUCT = "meta-llama/Llama-3.1-70B-Instruct"
ROBERTA_SQUAD_v2 = "deepset/roberta-base-squad2"
torch.manual_seed(42)

micro_models = {LLAMA_3p1_8B_INSTRUCT, GRANITE_3p2_8B_INSTRUCT, GRANITE_20B_CODE_INSTRUCT_8K, LLAMA_3p1_70B_INSTRUCT}


class AIUModelFixtureMixin(ModelFixtureMixin):
    @pytest.fixture(scope="class", autouse=True)
    def uninitialized_model(self, model_id):
        if model_id in micro_models:
            get_model_kwargs = {"architecture": "hf_configured", "nlayers": 3}
        else:
            get_model_kwargs = {"architecture": "hf_pretrained"}

        aiu_model = get_model(
            variant=model_id,
            device_type="cpu",
            fused_weights=False,
            data_type=torch.float16,
            **get_model_kwargs,
        )

        return aiu_model

    @pytest.fixture(scope="class", autouse=True)
    def model(self, uninitialized_model):
        # we want to use reset parameter initialization here rather than the default random initialization
        uninitialized_model.eval()
        torch.set_grad_enabled(False)
        uninitialized_model.compile(backend="sendnn")
        return uninitialized_model


decoder_models = [LLAMA_3p1_8B_INSTRUCT, GRANITE_3p2_8B_INSTRUCT, GRANITE_20B_CODE_INSTRUCT_8K, LLAMA_3p1_70B_INSTRUCT]


class TestAIUDecoderModels(
    ModelConsistencyTestSuite,
    AIUModelFixtureMixin,
):
    # x is the main parameter for this model which is the input tensor
    _get_signature_params = ["x"]
    _get_signature_input_ids, _get_signature_optional_params = pad_input_ids(
        [torch.arange(start=5, end=65, dtype=torch.int64)], min_pad_length=64
    )

    @pytest.fixture(scope="class", autouse=True, params=decoder_models)
    def model_id(self, request):
        return request.param

    def test_model_unfused(self, model, signature):
        pytest.skip("All AIU models are already unfused")


tuple_output_models = [ROBERTA_SQUAD_v2]


class TestAIUModelsTupleOutput(
    ModelConsistencyTestSuite,
    AIUModelFixtureMixin,
):
    # x is the main parameter for this model which is the input tensor
    _get_signature_params = ["x"]
    _get_signature_input_ids, _get_signature_optional_params = pad_input_ids(
        [torch.arange(start=5, end=65, dtype=torch.int64)],
        min_pad_length=64,
        is_causal_mask=False,
    )

    @pytest.fixture(scope="class", autouse=True, params=tuple_output_models)
    def model_id(self, request):
        return request.param

    @staticmethod
    def _get_signature_logits_getter_fn(f_out) -> torch.Tensor:
        return torch.cat([f_out[0], f_out[1]], dim=-1)

    def test_model_unfused(self, model, signature):
        pytest.skip("All AIU models are already unfused")
