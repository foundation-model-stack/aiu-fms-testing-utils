# Standard
from functools import partial
from typing import Any
import argparse
import json
import os

# Third Party
import torch
from torch import nn

# Local Packages
from aiu_fms_testing_utils.utils.aiu_setup import dprint


def import_addons(args: argparse.Namespace) -> None:
    """Import addons from FMS-MO. The import operation will register the selected
    quantization addon (comprising adapter, linear module, and custom op) with FMS.
    """

    try:
        if args.quantization == "gptq" and "aiu" in args.device_type:
            from fms_mo.aiu_addons.gptq import gptq_aiu_adapter, gptq_aiu_linear  # noqa: F401
        elif args.quantization == "fp8":
            from fms_mo.aiu_addons.fp8 import fp8_adapter, fp8_attn, fp8_linear  # noqa: F401
        elif args.quantization == "int8":
            from fms_mo.aiu_addons.i8i8 import i8i8_aiu_adapter, i8i8_aiu_linear  # noqa: F401
        dprint("Loaded `aiu_addons` functionalities")
    except ImportError:
        raise ImportError(f"Failed to import {args.quantization} addons from FMS-MO.")


def get_linear_config(args: argparse.Namespace) -> dict[str, Any]:
    """Return a linear_config dictionary to be used to instantiate quantized modules
    by FMS get_model
    """

    fused_weights = not args.unfuse_weights
    if args.quantization == "gptq":
        if fused_weights and args.is_aiu_backend:
            raise ValueError(
                "GPTQ checkpoints on AIU must always run with --unfuse_weights"
            )
        if args.default_dtype is not None:
            raise ValueError(
                "GPTQ default_dtype must be None to preserve the checkpoint data types."
            )

        if "aiu" in args.device_type:
            linear_type = "gptq_aiu"
        elif args.device_type == "cpu":
            linear_type = "gptq_cpu"
        elif args.device_type == "cuda":
            linear_type = "gptq"  # GPTQ support on GPU is FMS-native
        else:
            raise ValueError(f"Unsupported device {args.device} for GPTQ")

        qconfig_path = args.model_path + "/quantize_config.json"
        if os.path.exists(qconfig_path):
            with open(qconfig_path, "r") as f:
                dprint(f"loading quantization config from {qconfig_path}")
                qconfig = json.load(f)
                group_size = qconfig["group_size"]
                desc_act = qconfig["desc_act"]
                if desc_act:
                    raise NotImplementedError(
                        "Activation reordering not supported at this time."
                    )
        else:
            dprint(
                "[WARNING] Could not locate quantization config file. "
                "Default configuration will be used."
            )
            group_size = 128
            desc_act = False

        linear_config = {
            "linear_type": linear_type,
            "group_size": group_size,
            "desc_act": desc_act,
        }
    elif args.quantization == "fp8":
        dprint(
            "[INFO] fp8 config is inferred from HF checkpoint via FMS / FMS-MO functions"
        )
        return None
    elif args.quantization == "int8":
        if fused_weights and args.is_aiu_backend:
            raise ValueError(
                "INT8 checkpoints on AIU must always run with --unfuse_weights"
            )
        if args.default_dtype is not None:
            raise ValueError(
                "INT8 default_dtype must be None to preserve the checkpoint data types."
            )

        def select_int8_module(
            module_name: str | None = None,
            smoothquant: bool = True,
            smoothquant_layers: list[str] | None = None,
        ):
            if module_name is None:
                return "int8_aiu"
            smoothquant_on_module = (
                any([m in module_name for m in smoothquant_layers])
                if smoothquant_layers is not None
                else True
            )
            use_smoothquant = smoothquant and smoothquant_on_module
            return "int8_smoothquant_aiu" if use_smoothquant else "int8_aiu"

        if args.int8_smoothquant:
            # TODO: load info from config saved during quantization
            if any(
                "granite" in p.lower() for p in [args.model_path, args.architecture]
            ):
                smoothquant_layers = ["key", "value", "w1", "wg"]
            elif any(
                "roberta" in p.lower() for p in [args.model_path, args.architecture]
            ):
                smoothquant_layers = ["query", "key", "value", "w1"]
            else:
                raise NotImplementedError(
                    "INT8 architecture does not support smoothquant."
                )
        else:
            smoothquant_layers = []

        linear_config = {
            "linear_type": partial(
                select_int8_module,
                smoothquant=args.int8_smoothquant,
                smoothquant_layers=smoothquant_layers,
            ),
            "weight_per_channel": args.int8_weight_per_channel,
            "activ_quant_type": args.int8_activ_quant_type,
        }
    else:
        linear_config = {"linear_type": "torch_linear"}
    return linear_config


def validate_quantization(model: nn.Module, args: argparse.Namespace) -> None:
    """Ensure compatibility of FP8 models with device-specific operations."""

    has_fp8_weights = False
    has_bf16_weights = False
    has_fp16_weights = False
    for param in model.parameters():
        if param.dtype == torch.float8_e4m3fn:
            has_fp8_weights = True
        elif param.dtype == torch.bfloat16:
            has_bf16_weights = True
        elif param.dtype == torch.float16:
            has_fp16_weights = True

    if has_fp8_weights:
        if args.is_aiu_backend and has_bf16_weights and not args.cast_bf16_to_fp16:
            raise ValueError(
                "FP8 checkpoints on AIU with bf16 weights require casting to fp16 "
                "using --cast_bf16_to_fp16. Do not use --default_dtype!"
            )
        elif (
            args.device.type == "cuda"
            and has_fp16_weights
            and not args.cast_fp16_to_bf16
        ):
            raise ValueError(
                "FP8 checkpoints on GPU with fp16 weights require casting to bf16 "
                "using --cast_fp16_to_bf16. Do not use --default_dtype!"
            )
