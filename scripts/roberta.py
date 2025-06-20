import os
import sys
import argparse
import tempfile

from aiu_fms_testing_utils.utils import aiu_setup
from aiu_fms_testing_utils.utils.aiu_setup import dprint, world_rank, world_size


# PyTorch
import torch
import torch.distributed

# HuggingFace Transformers
from transformers import (
    AutoModelForMaskedLM,
    RobertaTokenizerFast,
    pipeline,
)

# TPEmbedding in FMS uses the torch.ops._c10d_functional.all_gather_into_tensor funciton
# which is not supported by GLOO. Eventhough we don't use GLOO in AIU execution, PyTorch
# doesn't know that and throws an error.
# This should be addressed in a future version of PyTorch, but for now disable it.
os.environ.setdefault("DISTRIBUTED_STRATEGY_IGNORE_MODULES", "WordEmbedding,Embedding")

# Foundation Model Stack
from fms.models import get_model
from fms.models.hf import to_hf_api

# Import AIU Libraries
from torch_sendnn import torch_sendnn

# ==============================================================
# Main
# ==============================================================
if __name__ == "__main__":
    # Number of batches to create
    NUM_BATCHES = 1

    # -------------
    # Command line argument parsing
    # -------------
    parser = argparse.ArgumentParser(
        description="PyTorch Small Toy Tensor Parallel Example"
    )
    parser.add_argument(
        "--backend",
        help="PyTorch Dynamo compiler backend",
        default="cpu",
        choices=["cpu", "aiu"],
    )
    pargs = parser.parse_args()

    if pargs.backend == "aiu":
        dynamo_backend = "sendnn"
    else:
        dynamo_backend = "inductor"

    is_distributed = world_size > 1
    if is_distributed:
        # Initialize the process group
        torch.distributed.init_process_group(
            backend="gloo", rank=world_rank, world_size=world_size
        )
        # Looks like a string compare, but is actually comparing the components
        # https://github.com/pytorch/pytorch/blob/b5be4d8c053e22672719b9a33386b071daf9860d/torch/torch_version.py#L10-L16
        if torch.__version__ < "2.3.0":
            # Fix until PyTorch 2.3
            torch._C._distributed_c10d._register_process_group(
                "default", torch.distributed.group.WORLD
            )

    # -------------
    # Setup AIU specific environment variables
    # -------------
    if "sendnn" in dynamo_backend:
        aiu_setup.aiu_dist_setup(world_rank, world_size)

    # -------------
    # Display some diagnostics
    # -------------
    if 0 == world_rank:
        dprint("-" * 60)
        dprint(
            f"Python Version  : {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        )
        dprint(f"PyTorch Version : {torch.__version__}")
        dprint(f"Dynamo Backend  : {pargs.backend} -> {dynamo_backend}")
        if pargs.backend == "aiu":
            for peer_rank in range(world_size):
                pcie_env_str = "AIU_WORLD_RANK_" + str(peer_rank)
                dprint(f"PCI Addr. for Rank {peer_rank} : {os.environ[pcie_env_str]}")
        print("-" * 60)
    if is_distributed:
        torch.distributed.barrier()

    # -------------
    # Create the model
    # -------------
    if 0 == world_rank:
        dprint(f"Creating the model...")
    # model_name = "roberta-base"
    # model_name = "deepset/roberta-base-squad2-distilled"
    model_name = "FacebookAI/roberta-base"
    hf_model = AutoModelForMaskedLM.from_pretrained(model_name)
    with tempfile.TemporaryDirectory() as workdir:
        hf_model.save_pretrained(
            f"{workdir}/roberta-base-masked_lm", safe_serialization=False
        )
        model = get_model(
            "roberta",
            "base",
            f"{workdir}/roberta-base-masked_lm",
            "hf",
            norm_eps=1e-5,
            tie_heads=True,
        )
    hf_model_fms = to_hf_api(
        model, task_specific_params=hf_model.config.task_specific_params
    )
    # hf_model_fms = get_model(
    #     architecture="hf_pretrained",
    #     variant=model_name
    # )

    # -------------
    # Compile the model
    # -------------
    if 0 == world_rank:
        dprint(f"Compiling the model...")
    the_compiled_model = torch.compile(hf_model_fms, backend=dynamo_backend)
    the_compiled_model.eval()  # inference only mode
    torch.set_grad_enabled(False)

    # -------------
    # Run the model
    # - First run the compiler will activate to create the artifacts
    # - Second run there is no compiler involved
    # -------------
    if is_distributed:
        torch.distributed.barrier()

    torch.manual_seed(42)
    tokenizer = RobertaTokenizerFast.from_pretrained(model_name)
    # prompt = "Hello I'm a <mask> model."
    # prompt = "Kermit the frog is a <mask>."
    prompt = "Miss Piggy is a <mask>."

    # First run will create compiled artifacts
    if 0 == world_rank:
        dprint(f"Running model: First Time...")
    unmasker = pipeline("fill-mask", model=the_compiled_model, tokenizer=tokenizer)
    the_output = unmasker(prompt)
    if 0 == world_rank:
        dprint(f"Answer: ({the_output[0]['score']:6.5f}) {the_output[0]['sequence']}")

    # Second run will be faster as it uses the cached artifacts
    if 0 == world_rank:
        dprint(f"Running model: Second Time...")
    unmasker = pipeline("fill-mask", model=the_compiled_model, tokenizer=tokenizer)
    the_output = unmasker(prompt)
    if 0 == world_rank:
        dprint(f"Answer: ({the_output[0]['score']:6.5f}) {the_output[0]['sequence']}")

    # -------------
    # Cleanup
    # -------------
    if 0 == world_rank:
        dprint(f"Done")
    if is_distributed:
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()
