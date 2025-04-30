from fms.models import get_model
import pytest
import torch

from fms.testing._internal.model_test_suite import (
    ModelConsistencyTestSuite,
    ModelFixtureMixin,
)
import os

if "HF_HOME" not in os.environ:
    os.environ["HF_HOME"] = "/tmp/models/hf_cache"

model_dir = os.environ.get("FMS_TESTING_MODEL_DIR", "/tmp/models")
LLAMA_194M = f"{model_dir}/llama-194m"
GRANITE_7B_BASE = f"{model_dir}/granite-7b-base"
GRANITE_8B_CODE_BASE = f"{model_dir}/granite-8b-code-base"
GRANITE_3_8B_CODE_BASE = f"{model_dir}/granite-3-8b-base"
GRANITE_20B_CODE_COBOL = f"{model_dir}/granite-20b-code-cobol-v1/20240603"

# models = [LLAMA_194M, GRANITE_7B_BASE, GRANITE_8B_CODE_BASE, GRANITE_3_8B_CODE_BASE]
models = [GRANITE_20B_CODE_COBOL]
mini_models = {LLAMA_194M, GRANITE_7B_BASE, GRANITE_8B_CODE_BASE, GRANITE_3_8B_CODE_BASE}

class AIUModelFixtureMixin(ModelFixtureMixin):

    @pytest.fixture(scope="class", autouse=True)
    def uninitialized_model(self, model_id):
        if model_id in mini_models:
            get_model_kwargs = {"architecture": "hf_configured", "nlayers": 3}
        else:
            get_model_kwargs = {"architecture": "hf_pretrained"}

        aiu_model = get_model(
            variant=model_id,
            device_type="cpu",
            unfuse_weights=True,
            **get_model_kwargs
        )
        torch.compile(aiu_model, backend="sendnn")
        return aiu_model

class TestAIUModels(
    ModelConsistencyTestSuite,
    AIUModelFixtureMixin,
):

    # x is the main parameter for this model which is the input tensor
    _get_signature_params = ["x"]

    @pytest.fixture(scope="class", autouse=True, params=models)
    def model_id(self, request):
        return request.param

    def test_model_unfused(self, model, signature):
        pytest.skip("All AIU models are already unfused")


ROBERTA_SQUAD_v2 = "deepset/roberta-base-squad2"
# tuple_output_models = [ROBERTA_SQUAD_v2]
tuple_output_models = [GRANITE_20B_CODE_COBOL]
class TestAIUModelsTupleOutput(
    ModelConsistencyTestSuite,
    AIUModelFixtureMixin,
):
    
    # x is the main parameter for this model which is the input tensor
    _get_signature_params = ["x"]

    @pytest.fixture(scope="class", autouse=True, params=tuple_output_models)
    def model_id(self, request):
        return request.param
    
    @staticmethod
    def _get_signature_logits_getter_fn(f_out) -> torch.Tensor:
        return torch.cat([f_out[0], f_out[1]], dim=-1)
    
    def test_model_unfused(self, model, signature):
        pytest.skip("All AIU models are already unfused")
    