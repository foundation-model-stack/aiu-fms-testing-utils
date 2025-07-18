# Standard
import argparse

# Local Packages
from aiu_fms_testing_utils.utils.aiu_setup import dprint


def get_args(parser: argparse.ArgumentParser) -> argparse.Namespace:
    # Arguments for FMS model loading
    args_model_loading = parser.add_argument_group("FMS model loading")
    args_model_loading.add_argument(
        "--architecture",
        type=str,
        help="The model architecture to benchmark.",
    )
    args_model_loading.add_argument(
        "--variant",
        type=str,
        default=None,
        help="The model variant (configuration) to benchmark (e.g., 7b, 13b, 70b).",
    )
    args_model_loading.add_argument(
        "--model_path",
        type=str,
        help="Path to the directory containing the model checkpoint(s).",
    )
    args_model_loading.add_argument(
        "--model_source",
        type=str,
        help="Source of the checkpoint. E.g. 'meta', 'hf', None",
    )
    args_model_loading.add_argument(
        "--unfuse_weights",
        action="store_true",
        help="If True, this will unfuse any fused weight modules.",
    )
    args_model_loading.add_argument(
        "--default_dtype",
        type=str,
        default=None,
        choices=["bf16", "fp16", "fp32"],
        help=(
            "If set to one of the choices, overrides the model checkpoint "
            "weight format by setting the default pytorch format."
        ),
    )
    parser.add_argument(
        "--cast_bf16_to_fp16",
        action="store_true",
        help=(
            "If set, cast any bf16 weights in the model to fp16 for AIU compiler. "
            "Doesn't touch fp32 or quantized."
        ),
    )
    parser.add_argument(
        "--cast_fp16_to_bf16",
        action="store_true",
        help=(
            "If set, cast any fp16 weights in the model to bf16 for GPU. "
            "Doesn't touch fp32 or quantized."
        ),
    )

    # Quantization arguments
    args_quantization = parser.add_argument_group("Model quantization")
    args_quantization.add_argument(
        "--quantization",
        type=str,
        choices=["gptq", "fp8"],
        default=None,
        help="Type of quantization of the model checkpoint",
    )

    # General run settings
    args_run_settings = parser.add_argument_group("Run settings")
    args_run_settings.add_argument(
        "--device_type",
        type=str,
        choices=["cuda", "cpu", "aiu", "aiu-senulator"],
        default="cuda",
        help="The device to run the model on",
    )
    args_run_settings.add_argument(
        "--seed",
        type=int,
        default=81072,
        help="Fix run seed for reproducibility",
    )
    args_run_settings.add_argument(
        "--output_path",
        type=str,
        default="",
        help="path of folder to save outputs to, if empty don't save",
    )
    args_run_settings.add_argument(
        "--tokenizer",
        type=str,
        required=True,
        help="Path to the tokenizer (e.g. ~/tokenizer.model)",
    )
    args_run_settings.add_argument(
        "--no_use_cache",
        action="store_false",
        help="Disable the kv-cache (on by default)",
    )
    args_run_settings.add_argument(
        "--deterministic",
        action="store_true",
        help=(
            "`deterministic` requires env variable `CUBLAS_WORKSPACE_CONFIG=:4096:8`"
            " when running on GPU. This flag is ignored on AIU."
        ),
    )
    args_run_settings.add_argument(
        "--distributed",
        action="store_true",
        help="This is a distributed job (multiple instances run with RANK+WORLD_SIZE)",
    )
    args_run_settings.add_argument(
        "-v", "--verbose", action="store_true", default=0, help="Enable verbose output"
    )

    # Arguments for compilation
    args_compile = parser.add_argument_group("Compiler")
    args_compile.add_argument(
        "--compile",
        action="store_true",
        help="Use torch.compile (slow for first inference pass)",
    )
    args_compile.add_argument(
        "--compile_mode",
        type=str,
        help="Mode for compilation (only valid for inductor backend)",
        default="default",
        choices=["default", "reduce-overhead"],
    )
    args_compile.add_argument(
        "--compile_backend",
        type=str,
        help="Backend for compilation (only when not running on AIU)",
        default="inductor",
        choices=["inductor", "eager", "aot_eager"],
    )
    args_compile.add_argument(
        "--compile_dynamic",
        action="store_true",
        help="Use dynamic shapes with torch.compile",
    )
    args_compile.add_argument(
        "--compile_dynamic_sendnn",
        action="store_true",
        help="Use dynamic shapes with aiu compile",
    )

    # Arguments shared between Decoder (future support) and Encoder models
    args_dec_enc = parser.add_argument_group("Decoders or Encoders (shared args)")
    args_dec_enc.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="size of input batch",
    )
    args_dec_enc.add_argument(
        "--max_prompt_length",
        type=int,
        default=None,
        help=(
            "Cap the number of tokens per prompt to a maximum length prior to padding. "
            "If None, prompts to decoder models will have no cap, while prompts to "
            "encoder models will be capped to a default of 384 tokens (for QA task)."
        ),
    )

    # Encoder model arguments
    args_encoder = parser.add_argument_group("Encoders")
    args_encoder.add_argument(
        "--dataset_name",
        type=str,
        default="squad_v2",
        help="The name of the dataset to use (via the datasets library).",
    )
    args_encoder.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The configuration name of the dataset to use (via the datasets library).",
    )
    args_encoder.add_argument(
        "--n_best_size",
        type=int,
        default=20,
        help="Total number of n-best predictions to generate.",
    )
    args_encoder.add_argument(
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
    args_encoder.add_argument(
        "--version_2_with_negative",
        type=bool,
        default=True,
        help="If true, some of the examples do not have an answer.",
    )
    args_encoder.add_argument(
        "--max_answer_length",
        type=int,
        default=30,
        help=(
            "The maximum length of an answer that can be generated. This is needed "
            "because the start and end predictions are not conditioned on one another."
        ),
    )
    args_encoder.add_argument(
        "--validation_file",
        type=str,
        default=None,
        help="A csv or a json file containing the validation data.",
    )
    args_encoder.add_argument(
        "--pad_to_max_length",
        action="store_true",
        help=(
            "If passed, pad all samples to `max_seq_length`. "
            "Otherwise, pad each batch individually to the longest sequence."
        ),
    )
    args_encoder.add_argument(
        "--max_eval_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of "
            "evaluation examples to this value if set."
        ),
    )
    args_encoder.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=1,
        help="Number of workers used during preprocessing of validation set (QA only).",
    )
    args_encoder.add_argument(
        "--overwrite_cache",
        action="store_true",
        help="Overwrite the cached training and evaluation sets.",
    )
    args_encoder.add_argument(
        "--doc_stride",
        type=int,
        default=128,
        help=(
            "When splitting up a long document into chunks how much stride "
            "to take between chunks."
        ),
    )
    args = parser.parse_args()

    # Add convenient arguments to parser
    args.is_quantized = args.quantization is not None
    args.is_aiu_backend = "aiu" in args.device_type
    args.dynamo_backend = "sendnn" if args.is_aiu_backend else "inductor"
    args.fused_weights = not args.unfuse_weights
    args.force_16b_dtype = args.cast_bf16_to_fp16 or args.cast_fp16_to_bf16

    if args.verbose:
        dprint("=" * 60)
        dprint(args)
        dprint("=" * 60)
    return args
