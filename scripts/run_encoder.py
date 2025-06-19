# Standard
import argparse
import time

# Third Party
from fms.models import get_model
from fms.utils import tokenizers
from torch import distributed, set_grad_enabled

# Local Packages
from aiu_fms_testing_utils.utils.aiu_setup import dprint, rank, world_size
from aiu_fms_testing_utils.utils.args_parsing import get_args
from aiu_fms_testing_utils.utils.encoders_utils import (
    wrap_encoder,
    run_encoder_eval_qa,
    run_encoder_eval_mlm,
)
from aiu_fms_testing_utils.utils.model_setup import setup_model, print_model_params
from aiu_fms_testing_utils.utils.quantization_setup import (
    import_addons,
    get_linear_config,
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

model_path = args.model_path
if args.int8_direct_quantization:
    save_path = None

    # !!! insert DQ for encoders here

    # if DQ is used, args.model_path represent FP16 ckpt but we need to load the
    # newly-created INT8 ckpt. Without DQ, args.model_path is the INT8 ckpt already.
    model_path = save_path

# Retrieve linear configuration (quantized or not) to instantiate FMS model
linear_config = get_linear_config(args)

if rank == 0:
    dprint("="*60)
    dprint(f"model_path={args.model_path}")
    dprint(f"{linear_config=}")
    dprint(f"fused_weights={args.fused_weights}")
    dprint(f"data_type={default_dtype}")
    dprint("="*60 + "\n")

dprint("Loading model...")
loading_model_start = time.time()
model = get_model(
    args.architecture,
    args.variant,
    model_path=model_path,
    device_type="cpu" if args.is_aiu_backend else args.device_type,
    data_type=default_dtype,
    source=args.model_source,
    distributed_strategy=dist_strat,
    group=distributed.group.WORLD,
    linear_config=linear_config,
    fused_weights=args.fused_weights,
)

if args.is_quantized:
    print_model_params(model, args)

tokenizer = tokenizers.get_tokenizer(args.tokenizer)

model.eval()
set_grad_enabled(False)
if args.distributed:
    distributed.barrier()
dprint(f"Loading model completed in {time.time() - loading_model_start:.2f} s.")

if args.architecture == "roberta":
    model = wrap_encoder(model)

if args.compile:
    dprint("Compiling model...")
    if args.is_aiu_backend:
        model.compile(backend="sendnn_decoder")
    else:
        # compiling can make first inference pass slow
        model.compile(mode=args.compile_mode, backend=args.compile_backend)
    dprint("Model compiled.")
else:
    dprint("[WARNING] SKIP COMPILE.")

if args.architecture == "roberta_question_answering":
    run_encoder_eval_qa(model, tokenizer, args)
elif args.architecture == "roberta":  # basic MaskedLM downstream task
    run_encoder_eval_mlm(model, tokenizer, args)

if args.distributed:
    distributed.barrier()
    distributed.destroy_process_group()
