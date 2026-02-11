import os
from typing import Any, Dict
import torch

from fms.models import get_model

from aiu_fms_testing_utils.utils import stagger_region
from aiu_fms_testing_utils.utils.aiu_setup import dprint
from aiu_fms_testing_utils.utils.dpp_config import DPPRunnerConfig
from aiu_fms_testing_utils.utils.env_utils import scoped_environ
from aiu_fms_testing_utils.testing.dpp.program_models import DeviceType


def _get_model_kwargs(model_variant: str) -> Dict[str, Any]:
    """Constructs model loading kwargs based on whether variant is a path or ID.

    Determines if the model_variant is a local filesystem path or a HuggingFace
    model identifier, and returns the appropriate keyword arguments for model loading.

    Args:
        model_variant: Either a local path to model files or a HuggingFace model ID.

    Returns:
        Dictionary with either "model_path" (for local paths) or "variant"
        (for HuggingFace IDs) as the key."""

    model_kwargs = {}
    if os.path.exists(model_variant):
        model_kwargs["model_path"] = model_variant
    else:
        model_kwargs["variant"] = model_variant

    return model_kwargs


def _prepare_fp8_weights(model: torch.nn.Module) -> None:
    """Converts model weights from bfloat16 to float16 for FP8 attention.

    When using FP8 attention variants, this function converts all bfloat16 parameters
    to float16. Issues a warning if any parameter values exceed the float16 range,
    which may cause accuracy loss.

    Args:
        model: PyTorch model whose weights may need conversion."""

    for name, param in model.named_parameters():
        if param.dtype == torch.bfloat16:
            if param.max() > torch.finfo(torch.float16).max:
                dprint(
                    f"[WARNING] You are casting param {name} to fp16, which will cause loss of accuracy."
                )
            param.data = param.data.to(dtype=torch.float16)


def load_model(
    device_type: DeviceType,
    is_fp8: bool,
    model_kwargs: Dict[str, Any],
    distributed_kwargs: Dict[str, Any],
    stagger_load: int,
    model_config: DPPRunnerConfig,
):
    """Loads and optionally compiles a model for inference or validation.

    Loads a model with the specified configuration. For Spyre/AIU models,
    compiles the model using the sendnn backend with dynamic compilation enabled.
    The scoped_environ context manager temporarily sets environment variables
    from model_config during compilation to configure the compiler's behavior (e.g.,
    program criteria, batch sizes, context lengths).

    Args:
        device_type: Target device for model execution.
        is_fp8: If True, uses FP8 quantization (dtype=None for auto-detection).
        model_kwargs: Dictionary with model loading parameters (variant or path).
        distributed_kwargs: Dictionary with distributed training configuration.
        stagger_load: Number of concurrent processes allowed during loading (0=unlimited).
        model_config: DPPRunnerConfig instance with environment variable updates.

    Returns:
        torch.nn.Module: Loaded model in evaluation mode. Spyre models are compiled
        with sendnn backend and may have FP8 weight conversion applied."""

    dtype = torch.float32 if device_type == DeviceType.CPU else torch.float16
    if is_fp8:
        dtype = None  # Let the model loading logic decide the appropriate FP8 dtype

    with stagger_region(stagger_load):
        model = get_model(
            architecture="hf_pretrained",
            device_type="cpu",
            data_type=dtype,
            fused_weights=False,
            **model_kwargs,
            **distributed_kwargs,
        )

    model.eval()

    if device_type != DeviceType.SPYRE:
        return model

    with scoped_environ(model_config.env_updates()):
        # Temporarily set environment variables needed for compile
        model.compile(backend="sendnn", options={"sendnn.dynamic": True})

    if is_fp8:
        _prepare_fp8_weights(model)

    return model
