import os
import sys
import argparse

from aiu_fms_testing_utils.utils import aiu_setup
from aiu_fms_testing_utils.utils.aiu_setup import dprint, world_rank, world_size

# PyTorch
import torch
import torch.distributed

# Foundation Model Stack
# - FeedForwardBlock : Building block for the model
# - apply_tp : Convert serial model into tensor parallel
from fms.modules.feedforward import FeedForwardBlock
from fms.utils.tp_wrapping import apply_tp

# Import AIU Libraries
from torch_sendnn import torch_sendnn  # noqa


# ==============================================================
# Toy Encoder Model
# ==============================================================
class ToyModelFM(torch.nn.Module):
    def __init__(self):
        super(ToyModelFM, self).__init__()
        # Input layer size
        self.INPUT_N = 1024
        # Hidden factor of the feedforward layer
        self.HIDDEN_FACTOR = 4
        # Number of feedforward layers
        self.LAYERS_N = 4
        self._linear_nets = torch.nn.ModuleList()
        for n in range(self.LAYERS_N):
            torch.manual_seed(42)
            block = FeedForwardBlock(
                self.INPUT_N,
                hidden_grow_factor=self.HIDDEN_FACTOR,
                activation_fn=torch.nn.ReLU(),
                p_dropout=0,
            )
            self._linear_nets.append(block)
        self._linear_nets.append(torch.nn.ReLU())

    def copy_weights(self, par_model, seq_model):
        self_parent_layer = self if par_model is None else par_model
        with torch.no_grad():
            for (seq_name, seq_layer), (self_name, self_layer) in zip(
                seq_model.named_children(), self_parent_layer.named_children()
            ):
                if hasattr(self_layer, "load_weights"):
                    self_layer.load_weights(
                        {
                            "w1.weight": seq_layer.w1.weight,
                            "w1.bias": seq_layer.w1.bias,
                            "w2.weight": seq_layer.w2.weight,
                            "w2.bias": seq_layer.w2.bias,
                        }
                    )
                else:
                    self.copy_weights(self_layer, seq_layer)

    def forward(self, x):
        _in = x
        for net in self._linear_nets:
            _in = net(_in)
        return _in


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
        dprint("Creating the model...")
    the_model = ToyModelFM()
    if is_distributed:
        # Create a Tensor Parallel version of the model
        apply_tp(the_model, torch.distributed.group.WORLD)

    # -------------
    # Compile the model
    # -------------
    if 0 == world_rank:
        dprint("Compiling the model...")
    the_compiled_model = torch.compile(the_model, backend=dynamo_backend)
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
    the_inputs = torch.randn(NUM_BATCHES, the_model.INPUT_N)

    # First run will create compiled artifacts
    if 0 == world_rank:
        dprint("Running model: First Time...")
    the_outputs = the_compiled_model(the_inputs)

    # Second run will be faster as it uses the cached artifacts
    if 0 == world_rank:
        dprint("Running model: Second Time...")
    the_outputs = the_compiled_model(the_inputs)

    # -------------
    # Cleanup
    # -------------
    if 0 == world_rank:
        dprint("Done")
    if is_distributed:
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()
