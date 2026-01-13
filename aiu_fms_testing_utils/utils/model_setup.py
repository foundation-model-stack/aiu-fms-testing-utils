# Standard
import argparse
import os
import sys

# Third party
import numpy as np
import random
import torch
from torch import nn, distributed

# Local
from aiu_fms_testing_utils.utils.aiu_setup import dprint, rank, local_rank, world_size
from aiu_fms_testing_utils.utils import aiu_setup


def get_default_dtype(args: argparse.Namespace) -> torch.dtype | None:
    """Return default_dtype for non-quantized models, otherwise None.
    If default_dtype is provided, it is set as torch default for non-quantized models.
    """

    default_dtype = None
    if not args.is_quantized:
        dtypes_map = {
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
            "fp32": torch.float32,
        }
        if args.default_dtype is not None:
            default_dtype = dtypes_map[args.default_dtype]
        if default_dtype is not None:
            torch.set_default_dtype(default_dtype)
    elif args.default_dtype is not None:
        raise ValueError(
            f"default_dtype (currently set to {args.default_dtype}) must be unset "
            "when running a quantized model."
        )
    return default_dtype


def get_device(args: argparse.Namespace) -> torch.device:
    """Return torch device and, if needed, set up AIU and its env variables.
    NOTE: args.device_type is str, but this function returns torch.device.
    """

    if args.device_type == "cuda":
        device = torch.device(args.device_type, local_rank)
        torch.cuda.set_device(device)
    elif args.is_aiu_backend:
        from torch_sendnn import torch_sendnn  # noqa: F401

        if args.distributed:
            aiu_setup.aiu_dist_setup(
                distributed.get_rank(),
                distributed.get_world_size(),
            )
        else:
            aiu_setup.aiu_setup(rank, world_size)
        aiu_setup.set_aiu_env_vars(args)
        device = torch.device("cpu")
    else:
        device = torch.device(args.device_type)
    return device


def print_system_setup(args: argparse.Namespace) -> None:
    """Display system info (rank 0 only)."""

    if args.verbose:
        dprint("-" * 60)
        dprint(
            f"Python Version  : {sys.version_info.major}."
            f"{sys.version_info.minor}.{sys.version_info.micro}"
        )
        dprint(f"PyTorch Version : {torch.__version__}")
        dprint(f"Dynamo Backend  : {args.device_type} -> {args.dynamo_backend}")
        dprint(f"Distributed     : {args.distributed}")
        if args.device_type == "aiu":
            for peer_rank in range(aiu_setup.world_size):
                pcie_env_str = "AIU_WORLD_RANK_" + str(peer_rank)
                dprint(f"PCI Addr. for Rank {peer_rank} : {os.environ[pcie_env_str]}")
        dprint("-" * 60)


def set_determinism(args: argparse.Namespace) -> None:
    """Set determinism.
    NOTE: torch determinism requires env variable: `CUBLAS_WORKSPACE_CONFIG=:4096:8`
    when running on GPU. This env variable is ignored on AIU.
    """

    if args.deterministic:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        torch.use_deterministic_algorithms(True)


def get_distributed_strategy(args: argparse.Namespace) -> str | None:
    """Return distributed strategy."""

    if args.distributed:
        dist_strat = "tp"
    else:
        if torch.cuda.device_count() > 1 and world_size == 1:
            dist_strat = "mp"
        else:
            dist_strat = None
    return dist_strat


def setup_model(args: argparse.Namespace) -> tuple[str | None, torch.device, str]:
    """Entry point for model setup."""

    default_dtype = get_default_dtype(args)
    device = get_device(args)
    print_system_setup(args)
    set_determinism(args)
    dist_strat = get_distributed_strategy(args)

    return default_dtype, device, dist_strat


def recast_16b(model: nn.Module, args: argparse.Namespace) -> None:
    """Cast 16-bit model parameters to selected datatype."""

    if args.cast_bf16_to_fp16:
        dprint(
            "Casting all BF16 model parameters to FP16 "
            "(--cast_bf16_to_fp16 flag is enabled)"
        )
        for name, param in model.named_parameters():
            if param.dtype == torch.bfloat16:
                if param.max() > torch.finfo(torch.float16).max:
                    dprint(
                        f"[WARNING] Casting param {name} to fp16 will truncate the "
                        "tensor. This may cause accuracy loss. Ignore this warning if "
                        "this is intended."
                    )
                param.data = param.data.to(dtype=torch.float16)
    elif args.cast_fp16_to_bf16:
        dprint(
            "Casting all FP16 model parameters to BF16 "
            "(--cast_fp16_to_bf16 flag is enabled)"
        )
        for param in model.parameters():
            if param.dtype == torch.float16:
                param.data = param.data.to(dtype=torch.bfloat16)


def print_model_params(model: nn.Module, args: argparse.Namespace) -> None:
    """Printout model and list of model parameters with related statistics."""

    if args.verbose:
        dprint("=" * 60 + "\n")
        dprint(
            "\n"
            + "\n".join(
                f"{k:70} {str(list(v.size())):15} {str(v.dtype):20} {str(v.device):10} "
                f"{v.float().min().item():12.4f} {v.float().max().item():12.4f}"
                for k, v in model.state_dict().items()
            )
        )
        dprint("=" * 60 + "\n")
    if args.architecture == "llama":
        dprint(
            "[NOTE] In Llama models, it's OK for bias and rotary embeddings to be "
            "marked as unused keys because of different architectural choices between "
            "FMS and HF models (but model output is preserved)."
        )
    dprint(model)
    dprint("=" * 60 + "\n")
