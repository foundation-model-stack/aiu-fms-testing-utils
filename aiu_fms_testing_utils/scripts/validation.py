import argparse
import json
import os
import random
import time
from pathlib import Path
import ast

import numpy as np
import torch
import torch._inductor.config
from fms.models import get_model, register_model
from fms.models.llama import LLaMAConfig, _llama_factory_factory
from fms.utils import generation
from fms.utils.generation import pad_input_ids
from torch import distributed as dist
from aiu_fms_testing_utils.utils import warmup_model
from aiu_fms_testing_utils.testing.validation import (
    LogitsExtractorHook,
    capture_level_1_metrics,
    extract_validation_information,
    GoldenTokenHook,
    filter_failed_level_1_cases,
    validate_level_0,
    load_validation_information,
    print_failed_cases,
)
from aiu_fms_testing_utils.utils import aiu_setup
from aiu_fms_testing_utils.utils.aiu_setup import dprint, rank, local_rank, world_size
from transformers import AutoTokenizer

# This example script validates models on AIU through comparisons to other devices.
parser = argparse.ArgumentParser(
    description="Script to validate AIU runs on causal models"
)
parser.add_argument(
    "--device_type",
    type=str,
    choices=["aiu", "aiu-senulator"],
    default="aiu",
    help="The device to run the model on",
)
parser.add_argument("--validation_device", type=str, default="cpu")
parser.add_argument(
    "--architecture",
    type=str,
    help="The model architecture to validate",
)
parser.add_argument(
    "--variant",
    type=str,
    default=None,
    help="The model variant (configuration) to validate. E.g. 7b, 13b, 70b.",
)
parser.add_argument(
    "--model_path",
    type=str,
    help="Path to the directory containing model weights",
)
parser.add_argument(
    "--model_source",
    type=str,
    help="Source of the checkpoint. E.g. 'meta', 'hf', None",
)
parser.add_argument(
    "--quantization",
    type=str,
    choices=["gptq"],
    default=None,
    help="Type of quantization of the model checkpoint",
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
    help="If set to one of the choices, overrides the model checkpoint weight format by setting the default pytorch format",
)
parser.add_argument(
    "--validation_compile",
    action="store_true",
    help="Use torch.compile for validation run",
)
parser.add_argument(
    "--compile_mode",
    type=str,
    help="Mode for validation compilation (only valid for inductor backend)",
    default="default",
    choices=["default", "reduce-overhead"],
)
parser.add_argument(
    "--compile_backend",
    type=str,
    help="Backend for validation compilation",
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
    "--validation_files_path",
    type=str,
    default="",
    help="if set, load the validated outputs from file(s) instead of locally generating them. Supports glob-style patterns",
)
parser.add_argument(
    "--validation_files_type",
    type=str,
    choices=["text", "tokens", "logits"],
    default="text",
    help="if validation_files_path is set, this informs the script of what kind of information we are loading for validation",
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
    "--validation_level",
    type=int,
    choices=[0, 1],
    default=0,
    help="Level at which you want to validate the AIU outputs. Level 0 is just token by token comparison for the outputs; Level 1 compares the logit outputs of the model against the validation logits, issues a warning if the cross entropy loss of the logits is above logits_loss_threshold, and replaces the token by the correct validated token to continue generation.",
)
parser.add_argument(
    "--logits_loss_threshold",
    type=float,
    default=2.5,
    help="Threshold at which to issue a warning because the logits are too different between validated run and AIU",
)
parser.add_argument(
    "--save_validation_info_path",
    type=str,
    default=None,
    help="If set, will save the validation info into the path specified for later use",
)
parser.add_argument(
    "--extra_get_model_kwargs",
    nargs="*",
    default={},
    help="Use this to override model configuration values to get model. Example: --extra_get_model_kwargs nlayers=2,...",
)
args = parser.parse_args()

dprint("*** DEPRECATION WARNING ***")
dprint("validation.py script has been deprecated in favor of test_decoders.py")

extra_get_model_kwargs = {}
for a in args.extra_get_model_kwargs:
    a_split = a.split("=")
    try:
        extra_get_model_kwargs[a_split[0]] = ast.literal_eval(a_split[1])
    except ValueError:
        extra_get_model_kwargs[a_split[0]] = a_split[1]

# this is a test model config
config = LLaMAConfig(
    emb_dim=1024,
    nheads=8,
    nlayers=10,
    src_vocab_size=128256,
)
register_model("llama", "194m", _llama_factory_factory(config))

dprint(f"{args}")

needs_validation_generation = args.validation_files_path == ""
needs_validation_forward = (
    not needs_validation_generation
    and args.validation_files_type in ["text", "tokens"]
    and args.validation_level == 1
)
needs_validation_run = needs_validation_forward or needs_validation_generation

fused_weights = not args.unfuse_weights

if args.quantization == "gptq":
    try:
        # validation script always loads AIU addon
        from fms_mo.aiu_addons.gptq import gptq_aiu_adapter, gptq_aiu_linear  # noqa: F401

        print("Loaded `aiu_addons` functionalities")

    except ImportError:
        print("Failed to import addon packages")
        raise Exception("GPTQ not enabled")

default_dtype = None
dtypes_map = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
}
if args.default_dtype is not None:
    default_dtype = dtypes_map[args.default_dtype]

if needs_validation_run:
    if args.validation_device == "cuda":
        validation_device = torch.device(args.validation_device, local_rank)
        torch.cuda.set_device(validation_device)
    else:
        validation_device = torch.device(args.validation_device)

if args.distributed:
    dist.init_process_group()
    # Fix until PT 2.3
    torch._C._distributed_c10d._register_process_group("default", dist.group.WORLD)
    aiu_setup.aiu_dist_setup(dist.get_rank(), dist.get_world_size())

# Always initialize AIU in this script
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
    os.environ["FLEX_COMPUTE"] = "SENTIENT"
    os.environ.setdefault("FLEX_DEVICE", "PF")

aiu_device = torch.device("cpu")
if default_dtype is not None:
    torch.set_default_dtype(default_dtype)

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

if args.quantization == "gptq":
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
        "linear_type": "gptq_aiu",
        "group_size": group_size,
        "desc_act": desc_act,
    }

    if needs_validation_run:
        if args.validation_device == "cpu":
            linear_type_validation = "gptq_cpu"
        elif args.validation_device == "cuda":
            linear_type_validation = "gptq"
        else:
            raise ValueError(
                f"Unsupported validation device {args.validation_device} for GPTQ"
            )
        linear_config_validation = {
            "linear_type": linear_type_validation,
            "group_size": group_size,
            "desc_act": desc_act,
        }
    # [ATTENTION] for GPTQ on AIU, we must always instantiate an unfused
    # model, the adapter will take care of converting key/values from
    # ckpt into the appropriate form for the model
    if fused_weights:
        raise ValueError(
            "GPTQ checkpoints on AIU must always run with --unfuse_weights"
        )
    default_dtype = None  # GPTQ dtype always comes from ckpt, can't be enforced
else:
    linear_config = {"linear_type": "torch_linear"}
    linear_config_validation = {"linear_type": "torch_linear"}

model = get_model(
    args.architecture,
    args.variant,
    model_path=args.model_path,
    device_type="cpu",
    data_type=default_dtype,
    source=args.model_source,
    distributed_strategy=distr_param,
    group=dist.group.WORLD,
    linear_config=linear_config,
    fused_weights=fused_weights,
    **extra_get_model_kwargs,
)

if args.quantization == "gptq":
    if args.architecture == "llama":
        dprint(
            "[NOTE] It's OK for unused keys to contain bias "
            "and rotary embeddings, in GPTQ LLaMA models"
        )
    dprint(model)
    dprint("=" * 60 + "\n")

if needs_validation_run:
    if args.quantization != "gptq":
        data_type_validation = (
            torch.float32 if validation_device == aiu_device else default_dtype
        )
    else:
        data_type_validation = default_dtype
    validation_model = get_model(
        args.architecture,
        args.variant,
        model_path=args.model_path,
        device_type=args.validation_device,
        data_type=data_type_validation,
        source=args.model_source,
        distributed_strategy=distr_param,
        group=dist.group.WORLD,
        linear_config=linear_config_validation,
        fused_weights=fused_weights,
        **extra_get_model_kwargs,
    )
    validation_model.load_state_dict(model.state_dict())
    if args.quantization == "gptq":
        if args.architecture == "llama":
            dprint(
                "[NOTE] It's OK for unused keys to contain bias and "
                "rotary embeddings, in GPTQ LLaMA models"
            )
        dprint(validation_model)
        dprint("=" * 60 + "\n")

tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
model.eval()
torch.set_grad_enabled(False)
loading_model_time = time.time() - loading_model_time
dprint(f"loading complete, took {loading_model_time:.3f}s")

dprint("compiling AIU model")
if not args.compile_dynamic:
    torch._dynamo.config.assume_static_by_default = True
    torch._dynamo.config.dynamic_shapes = False
    torch._dynamo.config.automatic_dynamic_shapes = False

model.compile(backend="sendnn")

if needs_validation_run and args.validation_compile:
    dprint("compiling validation model")
    validation_model.compile(mode=args.compile_mode, backend=args.compile_backend)

add_special_tokens = tokenizer.bos_token_id != tokenizer.eos_token_id


def truncate_prompts_to_max_length(prompts, max_len, max_allowed_length):
    # we may want the prompt length to be fixed to some max length
    # this will ensure that prior to padding the input ids
    if max_allowed_length is not None and max_len > max_allowed_length:
        dprint(f"max prompt length is {max_len}, truncating to {max_allowed_length}")
        prompts = [prompt[:max_allowed_length] for prompt in prompts]
    return prompts


### Load/set up prompts for generation
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
    ids, padding_kwargs = pad_input_ids(prompts, min_pad_length=padding_length)
else:
    ids = prompts
    padding_kwargs = {}


def print_result(result, result_idx: int = 0, file_prefix: str = ""):
    if local_rank != 0:
        return

    result = torch.tensor(result)
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
            file_path = output_path / f"{file_prefix}{result_idx}.txt"
            with file_path.open("w", encoding="utf-8") as file:
                file.write(output_str + "\n")

    dprint(output_str)
    print()


### Generate/load validation information
if not needs_validation_generation:
    validation_info = load_validation_information(
        args.validation_files_path,
        args.validation_files_type,
        args.batch_size,
        tokenizer,
    )

    val_tokens = [torch.tensor(_) for _ in validation_info.get_info("tokens")]
    max_val_len = max([prompt.size(0) for prompt in val_tokens])
    val_num_gen_tokens = int(args.max_new_tokens)
    if max_allowed_length is not None:
        val_tokens = truncate_prompts_to_max_length(
            val_tokens, max_val_len, max_allowed_length + val_num_gen_tokens
        )

    # Truncate each answer to its prompt length + max_new_tokens
    for i, prompt in enumerate(prompts):
        prompt_len = prompt.size(0)
        val_tokens[i] = val_tokens[i][: prompt_len + val_num_gen_tokens]

    if has_padding:
        val_ids, padding_val_kwargs = pad_input_ids(
            val_tokens, min_pad_length=val_num_gen_tokens + padding_length
        )
    else:
        val_ids = val_tokens
        padding_val_kwargs = None

    val_ids_list = torch.unbind(val_ids)
    for idx, validation_dict in enumerate(validation_info):
        validation_dict["tokens"] = val_ids_list[idx]

    if needs_validation_run:
        val_ids = val_ids.to(validation_device)
        if padding_val_kwargs is not None:
            for k in padding_val_kwargs.keys():
                padding_val_kwargs[k] = padding_val_kwargs[k].to(validation_device)

        if args.validation_device == "cpu":
            # Bug in 2.3.1 fixed in 2.4.1 for SDPA flash cpu impl when padding too much
            padding_val_kwargs["attn_algorithm"] = "math"

        val_logits = torch.unbind(
            validation_model(val_ids, **padding_val_kwargs).to("cpu")
        )

        for idx, validation_dict in enumerate(validation_info):
            if validation_dict["logits"] is None:
                # Generate the logits by running the model forward pass
                validation_dict["logits"] = val_logits[idx]
else:
    validation_info = extract_validation_information(
        validation_model,
        ids.to(validation_device),
        args.max_new_tokens,
        LogitsExtractorHook(),
        attn_algorithm="math",
        **padding_kwargs,
    )

warmup_model(
    model, ids, args.max_new_tokens, args.compile_dynamic_sendnn, **padding_kwargs
)

### AIU generation loop
static_tokens = validation_info.get_info("tokens")
post_iteration_hook = None
if args.validation_level >= 1:
    post_iteration_hook = GoldenTokenHook(static_tokens)

aiu_validation_info = extract_validation_information(
    model,
    ids,
    args.max_new_tokens,
    post_iteration_hook,
    eos_token_id=None if args.no_early_termination else tokenizer.eos_token_id,
    last_n_tokens=1,
    timing=args.timing,
    **padding_kwargs,
)

if args.save_validation_info_path is not None:
    validation_info.save(os.path.join(args.save_validation_info_path, "cpu"))
    aiu_validation_info.save(os.path.join(args.save_validation_info_path, "aiu"))

aiu_static_tokens = aiu_validation_info.get_info("tokens")
if args.validation_level == 0:
    failed_cases = validate_level_0(aiu_static_tokens, static_tokens)
else:
    level_1_metrics = capture_level_1_metrics(
        validation_info.get_info("logits"), aiu_validation_info.get_info("logits")
    )

    failed_cases = filter_failed_level_1_cases(
        level_1_metrics, lambda m: m >= args.logits_loss_threshold
    )

validation_passed = len(failed_cases) == 0

for i in range(len(aiu_static_tokens)):
    dprint("AIU text:")
    print_result(aiu_static_tokens[i], i, file_prefix="aiu_")
    dprint("Validation text:")
    print_result(static_tokens[i], i, file_prefix="val_")

if validation_passed:
    if args.validation_level == 0:
        dprint("The validation has passed! All the outputs match within threshold")
    elif args.validation_level == 1:
        dprint(
            "The validation has passed! All the logits cross entropy losses are below the threshold"
        )
else:
    if args.validation_level == 0:
        dprint("The validation has failed! There are some mismatched outputs")
    elif args.validation_level == 1:
        dprint(
            "The validation has failed! There are some logits cross entropy losses above the threshold"
        )
    print_failed_cases(failed_cases, aiu_static_tokens, static_tokens, tokenizer)
