# Standard
import argparse
import datetime
from functools import partial
import itertools
import json
import os
from pathlib import Path
import random
import time

# Third Party
from aiu_fms_testing_utils.utils import aiu_setup, warmup_model, stagger_region
from aiu_fms_testing_utils.utils.aiu_setup import dprint, rank, local_rank, world_size
import numpy as np
import torch
from torch import distributed as dist
from torch.fx.experimental import _config as fx_config
from fms.models import get_model, register_model
from fms.models.llama import LLaMAConfig, _llama_factory_factory
from fms.utils import generation
from fms.utils.generation import pad_input_ids

from transformers import AutoTokenizer


# This example script validates the LLaMA implementation by running inference on a couple of prompts.
#
# Example usage with single-GPU 7B model on slurm, with torch.compile and determinstic behavior:
# CUBLAS_WORKSPACE_CONFIG=:4096:8 srun -N 1 --gres=gpu:1 python scripts/inference.py --model_path=~/models/7B-F/ --tokenizer=~/models/tokenizer.model --compile --deterministic
# Example usage of 13B model on 2 GPUs with Tensor Parallel:
# srun -N 1 --gres=gpu:2 torchrun --nproc_per_node=2 scripts/inference.py --model_path=~/models/13B-F --tokenizer=~/models/tokenizer.model --distributed

parser = argparse.ArgumentParser(
    description="Script to run inference on a causal model"
)
parser.add_argument(
    "--device_type",
    type=str,
    choices=["cuda", "cpu", "aiu", "aiu-senulator"],
    default="cuda",
    help="The device to run the model on",
)
parser.add_argument(
    "--architecture",
    type=str,
    help="The model architecture to benchmark",
)
parser.add_argument(
    "--variant",
    type=str,
    default=None,
    help="The model variant (configuration) to benchmark. E.g. 7b, 13b, 70b.",
)
parser.add_argument(
    "--model_path",
    type=str,
    help="Path to the directory containing LLaMa weights (.pth files sharded by tensor parallel rank, not HF weights)",
)
parser.add_argument(
    "--model_source",
    type=str,
    help="Source of the checkpoint. E.g. 'meta', 'hf', None",
)
parser.add_argument(
    "--quantization",
    type=str,
    choices=["gptq", "int8"],
    default=None,
    help="Type of quantization of the model checkpoint",
)
parser.add_argument(
    "--int8_weight_per_channel",
    action="store_true",
    help="Enable per-channel weight quantization in INT8 quantized model",
)
parser.add_argument(
    "--int8_activ_quant_type",
    default="per_token",
    choices=["per_token", "per_tensor_symm", "per_tensor_asymm"],
    type=str,
    help="Define strategy for activation quantization in INT8 quantized model",
)
parser.add_argument(
    "--int8_smoothquant",
    action="store_true",
    help="Enable smoothquant in INT8 quantized model",
)
parser.add_argument(
    "--tokenizer",
    type=str,
    required=True,
    help="Path to the tokenizer (e.g. ~/tokenizer.model)",
)
parser.add_argument(
    "--no_use_cache",
    action="store_false",
    help="Disable the kv-cache (on by default)",
)
parser.add_argument(
    "--unfuse_weights",
    action="store_true",
    help="If set to True, this will unfuse any fused weight modules that support the unfuse_weights method",
)
parser.add_argument(
    "--default_dtype",
    type=str,
    default=None,
    choices=["bf16", "fp16", "fp32"],
    help="If set to one of the choices, overrides the model checkpoint weight format by setting the default pytorch format. This will break quantized checkpoints.",
)
parser.add_argument(
    "--cast_bf16_to_fp16",
    action="store_true",
    help="If set, cast any bf16 weights in the model to fp16 for AIU compiler. Doesn't touch fp32 or quantized",
)
parser.add_argument(
    "--cast_fp16_to_bf16",
    action="store_true",
    help="If set, cast any fp16 weights in the model to bf16 for GPU. Doesn't touch fp32 or quantized",
)
parser.add_argument(
    "--compile",
    action="store_true",
    help="Use torch.compile (slow for first inference pass)",
)
parser.add_argument(
    "--compile_mode",
    type=str,
    help="Mode for compilation (only valid for inductor backend)",
    default="default",
    choices=["default", "reduce-overhead"],
)
parser.add_argument(
    "--compile_backend",
    type=str,
    help="Backend for compilation (only when not running on AIU)",
    default="inductor",
    choices=["inductor", "eager", "aot_eager"],
)
parser.add_argument(
    "--compile_dynamic",
    action="store_true",
    help="Use dynamic shapes with torch.compile",
)
parser.add_argument(
    "--compile_dynamic_sendnn",
    action="store_true",
    help="Use dynamic shapes with aiu compile",
)
parser.add_argument(
    "--deterministic",
    action="store_true",
    help="Set torch.use_deterministic_algorithms? Requires env variable `CUBLAS_WORKSPACE_CONFIG=:4096:8`",
)
parser.add_argument(
    "--distributed",
    action="store_true",
    help="This is a distributed job (multiple instances run with RANK+WORLD_SIZE)",
)
parser.add_argument(
    "--batch_size",
    type=int,
    default=1,
    help="size of input batch",
)
parser.add_argument(
    "--max_prompt_length",
    type=int,
    default=None,
    help="cap the number of tokens per prompt to a maximum length prior to padding. If None, there will be no cap.",
)
parser.add_argument(
    "--min_pad_length",
    type=int,
    help="Pad inputs to a minimum specified length. If any prompt is larger than the specified length, padding will be determined by the largest prompt",
    default=0,
)
parser.add_argument(
    "--fixed_prompt_length",
    type=int,
    help="If defined, overrides both min_pad_length and max_prompt_length. Pads input to fixed_prompt_length, fails if any input needs truncation.",
    default=0,
)
parser.add_argument(
    "--max_new_tokens",
    type=int,
    help="max number of generated tokens",
    default=100,
)
parser.add_argument(
    "--no_early_termination",
    action="store_true",
    help="disable early termination on generation",
)
parser.add_argument(
    "--prompt_type",
    type=str,
    choices=["chat", "code"],
    default="chat",
    help="type of prompts to be used, either chat or code",
)
parser.add_argument(
    "--prompt_path",
    type=str,
    default="",
    help="if set, load the prompts from file(s) instead of the local examples. Supports glob-style patterns",
)
parser.add_argument(
    "--output_path",
    type=str,
    default="",
    help="path of folder to save outputs to, if empty don't save",
)
parser.add_argument(
    "--timing",
    type=str,
    choices=["e2e", "per-token"],
    default="",
    help="if set, how to time the generation of tokens, e2e or per-token",
)
parser.add_argument(
    "--iters",
    type=int,
    default=1,
    help="Number of iterations of inference to perform. Used for variance performance capture.",
)
parser.add_argument(
    "-v",
    "--verbose",
    action="count",
    default=0,
    help="Set verbosity level (pass flag as `-v`, `-vv`, `-vvv`)",
)
parser.add_argument(
    "--attention_type",
    type=str,
    choices=["sdpa", "paged", "math_fp8", "paged_fp8"],
    default="sdpa",
    help="which backend attention to use in mha",
)
parser.add_argument(
    "--prefill_chunk_size",
    type=int,
    default=0,
    help="if > 0, activate chunked prefill, with chunk_size=this_argument. Only works with paged attention variants",
)
parser.add_argument(
    "--stagger_load",
    type=int,
    default=0,
    help="Limit the number of concurrent processes executing the model loading phase. Set to 0 to allow all processes",
)
parser.add_argument(
    "--stagger_update_lazyhandle",
    type=int,
    default=0,
    help="Limit the number of concurrent processes executing the AIU update_lazyhandle phase. Set to 0 to allow all processes",
)
parser.add_argument(
    "--dist_timeout",
    type=int,
    default=0,
    help="Timeout to use for messaging in minutes. Default set by PyTorch dist.init_process_group",
)
args = parser.parse_args()

attention_map = {
    "sdpa": "sdpa_causal",
    "paged": "spyre_paged_attn",
    "math_fp8": "math_fp8",
    "paged_fp8": "spyre_paged_attn_fp8",
}

attn_name = attention_map[args.attention_type]

if "paged" in attn_name:
    from aiu_fms_testing_utils.utils.paged import generate
else:
    from fms.utils.generation import generate

if "fp8" in attn_name:
    import fms_mo.aiu_addons.fp8.fp8_attn  # noqa: F401

if args.quantization == "gptq":
    if "aiu" in args.device_type:
        try:
            from fms_mo.aiu_addons.gptq import gptq_aiu_adapter, gptq_aiu_linear  # noqa

            print("Loaded `aiu_addons` functionalities")
        except ImportError:
            raise ImportError("Failed to import GPTQ addons from fms-mo.")
elif args.quantization == "int8":
    try:
        from fms_mo.aiu_addons.i8i8 import i8i8_aiu_adapter, i8i8_aiu_linear  # noqa

        print("Loaded `aiu_addons` functionalities")
    except ImportError:
        raise ImportError("Failed to import INT8 addons from fms-mo.")

# this is a test model config
config = LLaMAConfig(
    emb_dim=1024,
    nheads=8,
    nlayers=10,
    src_vocab_size=128256,
)
register_model("llama", "194m", _llama_factory_factory(config))

default_dtype = None
dtypes_map = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
}
if args.default_dtype is not None:
    default_dtype = dtypes_map[args.default_dtype]

if default_dtype is not None:
    torch.set_default_dtype(default_dtype)

dprint(f"{args}")

is_aiu_backend = "aiu" in args.device_type

if args.distributed:
    if args.dist_timeout > 0:
        # Default timeout:
        # https://docs.pytorch.org/docs/stable/distributed.html#torch.distributed.init_process_group
        dist.init_process_group(timeout=datetime.timedelta(minutes=args.dist_timeout))
        dprint(f"NOTICE: init_process_group timeout set to {args.dist_timeout} minutes")
    else:
        dist.init_process_group()
    # Fix until PT 2.3
    torch._C._distributed_c10d._register_process_group("default", dist.group.WORLD)
    aiu_setup.aiu_dist_setup(dist.get_rank(), dist.get_world_size())

if args.device_type == "cuda":
    device = torch.device(args.device_type, local_rank)
    torch.cuda.set_device(device)
elif is_aiu_backend:
    from torch_sendnn import torch_sendnn  # noqa

    if not args.distributed:
        aiu_setup.aiu_setup(rank, world_size)

    _target_cache_size = max(
        int(args.max_new_tokens * 2),
        int(args.min_pad_length * 2.5),
        int(args.fixed_prompt_length * 2.5),
    )
    _prompt_size = max(int(args.min_pad_length), int(args.fixed_prompt_length))
    if hasattr(torch._dynamo.config, "accumulated_cache_size_limit"):
        if _target_cache_size > torch._dynamo.config.accumulated_cache_size_limit:
            _prev = torch._dynamo.config.accumulated_cache_size_limit
            torch._dynamo.config.accumulated_cache_size_limit = _target_cache_size
            dprint(
                f"NOTICE: Adjusting torch._dynamo.config.accumulated_cache_size_limit from {_prev} to {torch._dynamo.config.accumulated_cache_size_limit} to accomodate prompt size of {_prompt_size} and decode tokens of {args.max_new_tokens}"
            )

    if _target_cache_size > torch._dynamo.config.cache_size_limit:
        _prev = torch._dynamo.config.cache_size_limit
        torch._dynamo.config.cache_size_limit = _target_cache_size
        dprint(
            f"NOTICE: Adjusting torch._dynamo.config.cache_size_limit from {_prev} to {torch._dynamo.config.cache_size_limit} to accomodate prompt size of {_prompt_size} and decode tokens of {args.max_new_tokens}"
        )

    if not args.compile_dynamic:
        torch._dynamo.config.assume_static_by_default = True
        torch._dynamo.config.dynamic_shapes = False
        torch._dynamo.config.automatic_dynamic_shapes = False

    # This should be set outside!!!
    os.environ.setdefault("SENCORES", "32")
    os.environ.setdefault("SENCORELETS", "2")
    os.environ.setdefault("DATA_PREC", "fp16")
    os.environ.setdefault("FLEX_OVERWRITE_NMB_FRAME", "1")
    os.environ.setdefault("DTCOMPILER_KEEP_EXPORT", "true")

    os.environ.setdefault("COMPILATION_MODE", "offline_decoder")

    if args.device_type == "aiu-senulator":
        os.environ["FLEX_COMPUTE"] = "SENULATOR"
        os.environ["FLEX_DEVICE"] = "MOCK"
    else:
        if "AIU_WORLD_RANK_0" not in os.environ:
            print("must set AIU_WORLD_RANK_0")
            exit()
        os.environ.setdefault("FLEX_COMPUTE", "SENTIENT")
        os.environ.setdefault("FLEX_DEVICE", "PF")

    device = torch.device("cpu")
else:
    device = torch.device(args.device_type)

# requires setting environment variable: `CUBLAS_WORKSPACE_CONFIG=:4096:8`
if args.deterministic:
    SEED = 42
    random.seed(SEED)
    torch.manual_seed(SEED)  # pytorch random seed
    np.random.seed(SEED)  # numpy random seed
    torch.use_deterministic_algorithms(True)

dprint("loading model")
loading_model_time = time.time()
if args.distributed:
    distr_param = "tp"
else:
    if torch.cuda.device_count() > 1 and world_size == 1:
        distr_param = "mp"
    else:
        distr_param = None

fused_weights = not args.unfuse_weights
if args.quantization == "gptq":
    if fused_weights and is_aiu_backend:
        raise ValueError(
            "GPTQ checkpoints on AIU must always run with --unfuse_weights"
        )
    if default_dtype is not None:
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
elif args.quantization == "int8":
    if fused_weights and is_aiu_backend:
        raise ValueError(
            "INT8 checkpoints on AIU must always run with --unfuse_weights"
        )
    if default_dtype is not None:
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
        # TODO: consider saving this info into config during quantization
        if any("granite" in p.lower() for p in [args.model_path, args.architecture]):
            smoothquant_layers = ["key", "value", "w1", "wg"]
        elif any("roberta" in p.lower() for p in [args.model_path, args.architecture]):
            smoothquant_layers = ["query", "key", "value", "w1"]
        else:
            raise NotImplementedError("INT8 architecture does not support smoothquant.")
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

dprint("=" * 60)
dprint(f"model_path={args.model_path}")
dprint(f"{linear_config=}")
dprint(f"{fused_weights=}")
dprint(f"data_type={default_dtype}")
dprint("=" * 60 + "\n")

with stagger_region(args.stagger_load):
    model = get_model(
        args.architecture,
        args.variant,
        model_path=args.model_path,
        device_type="cpu" if is_aiu_backend else args.device_type,
        data_type=default_dtype,
        source=args.model_source,
        distributed_strategy=distr_param,
        group=dist.group.WORLD,
        linear_config=linear_config,
        fused_weights=fused_weights,
    )

### Quantization

# FP8 model checks
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
    if is_aiu_backend and has_bf16_weights and not args.cast_bf16_to_fp16:
        raise ValueError(
            "FP8 checkpoints on AIU with bf16 weights require casting to fp16 using --cast_bf16_to_fp16. Do not use --default_dtype!"
        )
    elif device.type == "cuda" and has_fp16_weights and not args.cast_fp16_to_bf16:
        raise ValueError(
            "FP8 checkpoints on GPU with fp16 weights require casting to bf16 using --cast_fp16_to_bf16. Do not use --default_dtype!"
        )

if args.cast_bf16_to_fp16:
    for name, param in model.named_parameters():
        if param.dtype == torch.bfloat16:
            if param.max() > torch.finfo(torch.float16).max:
                dprint(
                    f"[WARNING] You are casting param {name} to fp16, which will cause loss of accuracy. You can ignore this warning if this is intended."
                )
            param.data = param.data.to(dtype=torch.float16)

if args.cast_fp16_to_bf16:
    for param in model.parameters():
        if param.dtype == torch.float16:
            param.data = param.data.to(dtype=torch.bfloat16)

if args.quantization in ["gptq", "int8"]:
    if rank == 0 and args.verbose > 0:
        dprint(
            "PARAMS:\n"
            + "\n".join(
                f"{k:60} {str(v.dtype):15} {str(v.device):10} {list(v.size())}"
                for k, v in model.named_parameters()
            )
        )
        dprint(
            "BUFFERS:\n"
            + "\n".join(
                f"{k:60} {str(v.dtype):15} {str(v.device):10} {list(v.size())}"
                for k, v in model.named_buffers()
            )
        )
        dprint("=" * 60 + "\n")
    if args.architecture == "llama":
        dprint(
            "[NOTE] In Llama models, it's OK for bias and rotary embeddings to be marked as unused keys."
        )
    dprint(model)
    dprint("=" * 60 + "\n")

tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
model.eval()
torch.set_grad_enabled(False)
loading_model_time = time.time() - loading_model_time
dprint(f"loading complete, took {loading_model_time:.3f}s")

if args.compile:
    dprint("compiling model")
    fx_config.backed_size_oblivious = "paged" in attn_name
    if is_aiu_backend:
        model.compile(
            backend="sendnn", options={"sendnn.dynamic": args.compile_dynamic_sendnn}
        )
    else:
        # compiling can make first inference pass slow
        model.compile(mode=args.compile_mode, backend=args.compile_backend)

add_special_tokens = tokenizer.bos_token_id != tokenizer.eos_token_id


def truncate_prompts_to_max_length(prompts, max_len, max_allowed_length):
    # we may want the prompt length to be fixed to some max length
    # this will ensure that prior to padding the input ids
    if max_allowed_length is not None and max_len > max_allowed_length:
        dprint(f"max prompt length is {max_len}, truncating to {max_allowed_length}")
        prompts = [prompt[:max_allowed_length] for prompt in prompts]
    return prompts


if args.prompt_path != "":
    # Before creating the Path object, check if prompt_path has a glob pattern
    if isinstance(args.prompt_path, str):
        prompt_path, sep, glob_pattern = args.prompt_path.partition("*")
    else:
        sep = ""
        glob_pattern = ""
    glob_pattern = sep + glob_pattern

    prompt_path = Path(os.path.expanduser(prompt_path))
    prompt_file_paths = []

    if prompt_path.is_dir():
        if glob_pattern != "":
            glob_pattern_list = [glob_pattern]
        else:
            glob_pattern_list = ["*.txt"]
        for glob_pattern_possibility in glob_pattern_list:
            file_list = list(prompt_path.glob(glob_pattern_possibility))
            if len(file_list) > 0:
                prompt_file_paths = sorted(file_list)
                break

    if prompt_path.is_file():
        prompt_file_paths = [prompt_path]

    # Check if we found some files
    assert len(prompt_file_paths) > 0, f"Can't find any prompt files at {prompt_path}"

    # Check if we have enough files
    assert len(prompt_file_paths) >= args.batch_size, (
        f"Not enough prompt files at {prompt_path} for a batch size of {args.batch_size}"
    )

    prompts = []
    for i, prompt_file_path in enumerate(prompt_file_paths):
        if i == args.batch_size:
            break
        prompts.append(
            tokenizer.encode(
                prompt_file_path.read_text(encoding="utf-8"), return_tensors="pt"
            ).squeeze(0)
        )

else:
    if args.prompt_type == "chat":
        template = "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{}\n\n### Response:"

        prompt1 = template.format(
            "Provide a list of instructions for preparing chicken soup."
        )
        prompt2 = template.format("Explain some popular greetings in Spanish.")
        prompt3 = template.format("Explain to me why ignorance is bliss.")
        prompt4 = template.format(
            "I have just come into a very large sum of money. Provide me a list of things that I can do with my new found wealth."
        )
    elif args.prompt_type == "code":
        template = "[INST] Write code to solve the following coding problem that obeys the constraints and passes the example test cases. Please wrap your code answer using ```:\n{}\n[/INST]"
        prompt1 = template.format("Write a bubble sort function in python.")
        prompt2 = template.format(
            "Using the Java streams API, write a simple function which will get the cumulative sum of a list of integers."
        )
        prompt3 = template.format(
            "In bash, how do I list all directories and sub-directories which contain a .py file."
        )
        prompt4 = template.format(
            "Write a simple decorator in python which will modify all string inputs to ints if possible."
        )
    else:
        dprint("prompt_type must be one of chat or code")
        exit()

    prompt1 = tokenizer.encode(prompt1, return_tensors="pt").squeeze(0)
    prompt2 = tokenizer.encode(prompt2, return_tensors="pt").squeeze(0)
    prompt3 = tokenizer.encode(prompt3, return_tensors="pt").squeeze(0)
    prompt4 = tokenizer.encode(prompt4, return_tensors="pt").squeeze(0)
    prompts = [prompt1, prompt2, prompt3, prompt4]
    prompts = prompts * ((args.batch_size // 4) + 1)
    prompts = prompts[: args.batch_size]

if args.fixed_prompt_length != 0:
    padding_length = args.fixed_prompt_length
    max_allowed_length = args.fixed_prompt_length
else:
    padding_length = args.min_pad_length
    max_allowed_length = args.max_prompt_length

has_padding = args.batch_size > 1 or padding_length != 0
max_len = max([len(prompt) for prompt in prompts])

if args.fixed_prompt_length != 0 and args.fixed_prompt_length < max_len:
    dprint(
        "One or more prompts require truncation. Truncation has been disabled as fixed_prompt_length has been set."
    )
    exit(1)
prompts = truncate_prompts_to_max_length(prompts, max_len, max_allowed_length)
if has_padding:
    ids, extra_generation_kwargs = pad_input_ids(prompts, min_pad_length=padding_length)
else:
    ids = prompts
    if isinstance(ids, list) and len(ids) == 1:
        ids = ids[0].unsqueeze(0)
    extra_generation_kwargs = {}

extra_generation_kwargs["attn_name"] = attn_name

if "paged" in attn_name:
    import bisect

    # the compiler supports certain max context lengths (VLLM_DT_MAX_CONTEXT_LEN)
    # this will ensure that we select smallest supported VLLM_DT_MAX_CONTEXT_LEN that fits the largest possible context (prompt size + max_new_tokens)
    # if the user provides their own VLLM_DT_MAX_CONTEXT_LEN, use this value instead
    __largest_context = ids.shape[1] + args.max_new_tokens
    __supported_context_lengths = [64, 128, 256, 512, 1024, 2048, 4096, 8192]
    os.environ.setdefault(
        "VLLM_DT_MAX_CONTEXT_LEN",
        str(
            __supported_context_lengths[
                bisect.bisect_left(__supported_context_lengths, __largest_context)
            ]
        ),
    )
    os.environ.setdefault("VLLM_DT_MAX_BATCH_SIZE", str(max(ids.shape[0], 2)))


def print_result(result, result_idx: int):
    if local_rank != 0:
        return
    if has_padding:
        result = generation.trim_prefix(result)

    result = generation.trim_prefix(result, tokenizer.bos_token_id)

    # stop at EOS token if present and remove padding
    if not args.no_early_termination:
        result = generation.truncate_after_eos(result, tokenizer.eos_token_id)

    output_str = tokenizer.decode(result)

    if args.output_path != "":
        output_path = Path(args.output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        if output_path.is_dir():
            file_path = output_path / f"{result_idx}.txt"
            with file_path.open("w", encoding="utf-8") as file:
                file.write(output_str + "\n")
    dprint(output_str)
    print()


def infer(use_cache, do_sample, warmup):
    # With greedy generation (do_sample=False) we _should_ always get the same results.
    # There is currently a bug in start_pos for batched rotary embeddings that can lead
    # varying results for the same prompt.
    if local_rank == 0 and not warmup:
        dprint(f"use_cache {use_cache};; do_sample {do_sample}")
        dprint("==================")

    # Add only_last_token optimization
    global extra_generation_kwargs
    if extra_generation_kwargs is None:
        extra_generation_kwargs = {}
    extra_generation_kwargs["last_n_tokens"] = 64 if "paged" in attn_name else 1

    if not args.no_early_termination and not warmup:
        eos_token_id = tokenizer.eos_token_id
    else:
        eos_token_id = None

    attention_specific_kwargs = {}
    if attn_name == "sdpa_causal":
        attention_specific_kwargs["contiguous_cache"] = True
        attention_specific_kwargs["max_seq_len"] = ids.shape[1] + args.max_new_tokens
    elif "paged" in attn_name:
        attention_specific_kwargs["prefill_chunk_size"] = args.prefill_chunk_size

    result = generate(
        model,
        ids,
        max_new_tokens=args.max_new_tokens,
        use_cache=use_cache,
        do_sample=do_sample,
        timing=args.timing,
        eos_token_id=eos_token_id,
        extra_kwargs=extra_generation_kwargs,
        **attention_specific_kwargs,
    )
    if args.timing != "":
        result, timings = result
        if args.timing == "e2e":
            dprint(f"E2E timing information: {timings[0]:.3f}s")
        elif args.timing == "per-token":
            if not warmup:
                dprint(f"First-token latency: {timings[0] * 1000:.3f} ms")
                dprint(
                    f"Average next-token latency (including first token): {np.mean(timings) * 1000:.3f} ms"
                )
                if len(timings) > 1:
                    dprint(
                        f"Average next-token latency: {np.mean(timings[1:]) * 1000:.3f} ms"
                    )
                    dprint(
                        f"Max next-token latency: {np.max(timings[1:]) * 1000:.3f} ms (token #{np.argmax(timings[1:]) + 2})"
                    )
                    dprint(
                        f"Min next-token latency: {np.min(timings[1:]) * 1000:.3f} ms (token #{np.argmin(timings[1:]) + 2})"
                    )
                    dprint(
                        f"Std deviation of next-token latencies: {np.std(timings[1:]) * 1000:.3f} ms"
                    )
            timings = [f"{t * 1000:.3f}" for t in timings]
            dprint(f"Per-token timing information: {', '.join(timings)} ms")
    if len(result.shape) == 1:
        result = result.unsqueeze(0)

    if not warmup:
        for i in range(result.shape[0]):
            print_result(result[i], i)


do_sample = [False]
use_cache = [
    args.no_use_cache
]  # True/False are identical with greedy iff `torch.use_deterministic_algorithms(True)`

if args.compile:
    dprint("compilation warmup")
    pt_compile_model_time = time.time()
    if "aiu" in args.device_type:
        for cache in use_cache:
            warmup_model(
                model,
                ids,
                args.max_new_tokens,
                args.compile_dynamic_sendnn,
                use_cache=cache,
                stagger_update_lazyhandle=args.stagger_update_lazyhandle,
                prefill_chunk_size=args.prefill_chunk_size,
                **extra_generation_kwargs,
            )
        if (
            args.device_type == "aiu"
        ):  # run device initialization warmup for AIU, skip for senulator
            aiu_warmup_time = time.time()
            for sample, cache in itertools.product(do_sample, use_cache):
                infer(cache, sample, True)
            aiu_warmup_time = time.time() - aiu_warmup_time
            dprint(
                f"AIU device initialization warmup complete, took {aiu_warmup_time:.3f}s"
            )
    else:
        for sample, cache in itertools.product(do_sample, use_cache):
            infer(cache, sample, True)
    pt_compile_model_time = time.time() - pt_compile_model_time
    dprint(f"PT compile complete, took {pt_compile_model_time:.3f}s")

dprint("generating output")

for sample, cache in itertools.product(do_sample, use_cache):
    for _ in range(args.iters):
        infer(cache, sample, False)
