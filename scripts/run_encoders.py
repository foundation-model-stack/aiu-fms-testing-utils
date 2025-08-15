# Standard
import argparse
import time

# Third Party
from fms.models import get_model
from fms.models.roberta import RoBERTaForQuestionAnswering, RoBERTa
from fms.models.hf.roberta.modeling_roberta_hf import HFAdaptedRoBERTaForMaskedLM
from torch import distributed, set_grad_enabled
from transformers import AutoTokenizer

# Local Packages
from aiu_fms_testing_utils.utils.aiu_setup import dprint, rank, world_size
from aiu_fms_testing_utils.utils.args_parsing import get_args
from aiu_fms_testing_utils.utils.encoders_utils import (
    wrap_encoder,
    run_encoder_eval_qa,
    run_encoder_eval_mlm,
)
from aiu_fms_testing_utils.utils.model_setup import (
    setup_model,
    print_model_params,
    recast_16b,
)
from aiu_fms_testing_utils.utils.quantization_setup import (
    import_addons,
    get_linear_config,
    validate_quantization,
)

parser = argparse.ArgumentParser(
    description="Entry point for AIU inference of encoder models."
)
args = get_args(parser)
args.is_encoder = True  # add argument directly into Namespace

if args.is_quantized:
    import_addons(args)

if args.distributed:
    distributed.init_process_group(backend="gloo", rank=rank, world_size=world_size)

# Main model setup
default_dtype, device, dist_strat = setup_model(args)
args.device = device

# Retrieve linear configuration (quantized or not) to instantiate FMS model
linear_config = get_linear_config(args)

dprint("=" * 60)
dprint(f"model_path={args.model_path}")
dprint(f"{linear_config=}")
dprint(f"fused_weights={args.fused_weights}")
dprint(f"data_type={default_dtype}")
dprint("=" * 60 + "\n")

dprint("Loading model...")
loading_model_start = time.time()
model = get_model(
    args.architecture,
    args.variant,
    model_path=args.model_path,
    device_type="cpu" if args.is_aiu_backend else args.device_type,
    data_type=default_dtype,
    source=args.model_source,
    distributed_strategy=dist_strat,
    group=distributed.group.WORLD,
    linear_config=linear_config,
    fused_weights=args.fused_weights,
)

if args.force_16b_dtype:
    recast_16b(model, args)

if args.is_quantized:
    validate_quantization(model, args)
    print_model_params(model, args)

tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

model.eval()
set_grad_enabled(False)
if args.distributed:
    distributed.barrier()
dprint(f"Loading model completed in {time.time() - loading_model_start:.2f} s.")

if isinstance(model, RoBERTa):
    model = wrap_encoder(model)  # enable using pipeline to eval RoBERTa MaskedLM

if args.compile:
    dprint("Compiling model...")
    if args.is_aiu_backend:
        model.compile(
            backend="sendnn",
            options={"sendnn.dynamic": args.compile_dynamic_sendnn},
        )
    else:
        # compiling can make first inference pass slow
        model.compile(mode=args.compile_mode, backend=args.compile_backend)
    dprint("Model compiled.")
else:
    dprint("Skip model compiling. Only for debug purpose.")

if isinstance(model, RoBERTaForQuestionAnswering):
    run_encoder_eval_qa(model, tokenizer, args)
elif isinstance(model, RoBERTa) or isinstance(model, HFAdaptedRoBERTaForMaskedLM):
    # basic MaskedLM downstream task
    run_encoder_eval_mlm(model, tokenizer, args)

if args.distributed:
    distributed.barrier()
    distributed.destroy_process_group()
