# Standard
import argparse

# Local Packages
from aiu_fms_testing_utils.utils.aiu_setup import dprint


def get_args(parser: argparse.ArgumentParser) -> argparse.Namespace:

    # FMS model loading arguments
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
        help=(
            "Path to the directory containing LLaMa weights "
            "(.pth files sharded by tensor parallel rank, not HF weights)"
        ),
    )
    parser.add_argument(
        "--model_source",
        type=str,
        help="Source of the checkpoint. E.g. 'meta', 'hf', None",
    )
    parser.add_argument(
        "--unfuse_weights",
        action="store_true",
        help=(
            "If set to True, this will unfuse any fused weight modules that "
            "support the unfuse_weights method"
        ),
    )
    parser.add_argument(
        "--default_dtype",
        type=str,
        default=None,
        choices=["bf16", "fp16", "fp32"],
        help=(
            "If set to one of the choices, overrides the model checkpoint "
            "weight format by setting the default pytorch format"
        ),
    )

    # Quantization arguments
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
    parser.add_argument(  # NOTE: roberta only so far but should expand to LLM
        "--direct_quantization",
        action="store_true",
        help="Train INT8 model with Direct Quantization",
    )
    parser.add_argument(
        "--num_dq_samples",
        type=int,
        default=128,
        help="number of samples used for Direct Quantization",
    )

    # General settings
    parser.add_argument(
        "--device_type",
        type=str,
        choices=["cuda", "cpu", "aiu", "aiu-senulator"],
        default="cuda",
        help="The device to run the model on"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=81072,
        help="Run seed (only needed if eval dataset is shuffled)",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="",
        help="path of folder to save outputs to, if empty don't save",
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
        "--deterministic",
        action="store_true",
        help="`deterministic` requires env variable `CUBLAS_WORKSPACE_CONFIG=:4096:8`",
    )
    parser.add_argument(
        "--distributed",
        action="store_true",
        help="This is a distributed job (multiple instances run with RANK+WORLD_SIZE)",
    )
    parser.add_argument(  # could be a bool / flag
        '-v', '--verbose',
        action='count',
        default=0,
        help="Set verbosity level (pass flag as `-v`, `-vv`, `-vvv`)"
    )

    # Compiling arguments
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

    # LLM-specific inference arguments
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
        help=(
            "Cap the number of tokens per prompt to a maximum length prior to padding. "
            "If None, there will be no cap."
        ),
    )
    parser.add_argument(
        "--min_pad_length",
        type=int,
        default=0,
        help=(
            "Pad inputs to a minimum specified length. If any prompt is larger than "
            "the specified length, padding will be determined by the largest prompt"
        ),
    )
    parser.add_argument(
        "--fixed_prompt_length",
        type=int,
        default=0,
        help=(
            "If defined, overrides both min_pad_length and max_prompt_length. "
            "Pads input to fixed_prompt_length, fails if any input needs truncation."
        ),
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
        help=(
            "If set, load the prompts from file(s) instead of the local examples. "
            "Supports glob-style patterns"
        ),
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
        help=(
            "Number of iterations of inference to perform. Used for variance "
            "performance capture."
        ),
    )

    # RoBERTa-specific evaluation arguments
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="squad_v2",
        help="The name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The configuration name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--n_best_size",
        type=int,
        default=20,
        help="Total number of n-best predictions to generate.",
    )
    parser.add_argument(
        "--null_score_diff_threshold",
        type=float,
        default=0.0,
        help=(
            "The threshold used to select the null answer: if the best answer has a "
            "score that is less than the score of the null answer minus this threshold, "
            "the null answer is selected for this example.  Only useful when "
            "`version_2_with_negative=True`."
        ),
    )
    parser.add_argument(
        "--version_2_with_negative",
        type=bool,
        default=True,
        help="If true, some of the examples do not have an answer.",
    )
    parser.add_argument(
        "--max_answer_length",
        type=int,
        default=30,
        help=(
            "The maximum length of an answer that can be generated. This is needed "
            "because the start and end predictions are not conditioned on one another."
        ),
    )
    parser.add_argument(
        "--validation_file",
        type=str,
        default=None,
        help="A csv or a json file containing the validation data.",
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=384,
        help=(
            "The maximum total input sequence length after tokenization. "
            "Sequences longer than this will be truncated, "
            "sequences shorter will be padded if `--pad_to_max_length` is passed."
        ),
    )
    parser.add_argument(
        "--pad_to_max_length",
        action="store_true",
        help=(
            "If passed, pad all samples to `max_seq_length`. "
            "Otherwise, dynamic padding is used."
        ),
    )
    parser.add_argument(
        "--max_eval_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of "
            "evaluation examples to this value if set."
        ),
    )
    parser.add_argument(
        "--preprocessing_num_workers", type=int, default=1, help=""
    )
    parser.add_argument(
        "--overwrite_cache",
        action="store_true",
        help="Overwrite the cached training and evaluation sets",
    )
    parser.add_argument(
        "--doc_stride",
        type=int,
        default=128,
        help=(
            "When splitting up a long document into chunks how much stride "
            "to take between chunks."
        ),
    )
    parser.add_argument(  # NOTE: consider replacing in code with batch_size (DQ vs eval?)
        "--per_device_eval_batch_size",
        type=int,
        default=1,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    args = parser.parse_args()

    # Add convenient arguments to parser
    args.is_encoder = "bert" in args.architecture.lower()  # TODO: improve this check
    args.is_quantized = args.quantization is not None
    args.is_aiu_backend = "aiu" in args.device_type
    args.dynamo_backend = "sendnn" if args.is_aiu_backend else "inductor"
    args.fused_weights = not args.unfuse_weights

    if args.verbose:
        dprint("=" * 60)
        dprint(args)
        dprint("=" * 60)
    return args
