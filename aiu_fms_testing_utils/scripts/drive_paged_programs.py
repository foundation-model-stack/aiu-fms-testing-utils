import argparse

from aiu_fms_testing_utils.testing.dpp.run_drive_paged_programs import run_dpp
from aiu_fms_testing_utils.testing.dpp.program_models import AttnType, TestType
from aiu_fms_testing_utils.utils.model_setup import Timing


def parse_cli_args() -> argparse.Namespace:
    """Initializes the argument parser and parses command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Script which will drive paged programs for debugging"
    )

    parser.add_argument(
        "--programs",
        metavar="N",
        type=str,
        nargs="*",
        default=[],
        help="""
        The list of programs to run. This would take a list where each element would be one of program_id OR <program_id>:<min_batch>,<min_prompt_length>.
        If program_id is specified any prompt that would result in this program would be selected.
        If <program_id>:<min_batch>,<min_prompt_length> is specified, then with the given program_id, select a prompt that satisfies min_batch and min_prompt_length (if none exists, a message will be printed to warn the user)
        If this list is empty, each program will be run once with any prompt that would result in this program being selected.
        """,
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=8,
        help="set this if you want to change the number of tokens generated per sequence (1 prefill + max_new_tokens-1 decodes). Note: If this value is larger than 64, this may result in switching decode programs mid generation",
    )
    parser.add_argument(
        "--distributed",
        action="store_true",
        help="This is a distributed job (multiple instances run with RANK+WORLD_SIZE)",
    )
    parser.add_argument(
        "--model_variant",
        type=str,
        default="ibm-ai-platform/micro-g3.3-8b-instruct-1b",
        help="The model id or path to use for this test. Note: must be a huggingface format",
    )
    parser.add_argument(
        "--timing",
        type=str,
        choices=["e2e", "per-token"],
        default="",
        help="if set, how to time the generation of tokens, e2e or per-token",
    )
    parser.add_argument(
        "--program_criteria_json_path",
        type=str,
        help="path to json file containing the program criteria list",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        help="path to dataset",
    )
    parser.add_argument(
        "--dataset_type",
        type=str,
        choices=["rag_factoid", "sharegpt", "custom"],
        default="sharegpt",
        help="selects the correct dataset type for sampling. Must be one of rag_factoid or sharegpt or custom",
    )
    parser.add_argument(
        "--test_type",
        type=str,
        choices=["tokens", "metrics"],
        default="metrics",
        help="set the type of the test that you would like to run. If metrics, will inject tokens and get metrics. If tokens, will not inject tokens and get tokens",
    )

    parser.add_argument(
        "--cross_entropy_threshold",
        type=float,
        default=2.5,
        help="threshold to denote passing/failing a given iteration",
    )

    parser.add_argument(
        "--failure_rate_threshold",
        type=float,
        default=0.1,
        help="the threshold which denotes whether to pass or fail the test. The failure threshold is defined as the number of failing iterations (cross_entropy) over the total iterations. If this value exceeds the failure_rate_threshold, we will fail the test",
    )

    parser.add_argument(
        "--attention_type",
        type=str,
        default="paged",
        choices=["paged", "paged_fp8"],
        help="The attention type to use",
    )
    parser.add_argument(
        "--prefill_chunk_size",
        type=int,
        default=0,
        help="if > 0, activate chunked prefill, with chunk_size=this_argument. Only works with paged attention variants.",
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
    parser.add_argument(
        "--skip_validation",
        action="store_true",
        help="set to true to skip cpu validation",
    )
    parser.add_argument(
        "--validation_info_outputs_dir",
        type=str,
        default="/home/senuser/models/validation_info",
        help="path to directory containing validation info outputs",
    )
    parser.add_argument(
        "--save_validation_info_outputs",
        action="store_true",
        help="set to true to save cpu validation outputs for later consumption",
    )
    parser.add_argument(
        "--prioritize_large_batch_sizes",
        action="store_true",
        help="set to true if you would like to prioritize large batch sizes",
    )
    parser.add_argument(
        "--enforce_homogeneous_prompt_programs",
        action="store_true",
        help="set to true ensure that all prompts hit the same prompt program for a given test",
    )

    return parser.parse_args()


def main() -> None:
    # Environment Setup
    args = parse_cli_args()

    attention_map = {
        "sdpa": "sdpa_causal",
        "paged": "spyre_paged_attn",
        "math_fp8": "math_fp8",
        "paged_fp8": "spyre_paged_attn_fp8",
    }

    try:
        attention_type = attention_map[args.attention_type]
    except KeyError:
        raise ValueError(
            f"Invalid attention type: {args.attention_type}. Expected one of {list(attention_map.keys())}."
        )

    run_dpp(
        programs=args.programs,
        dataset_path=args.dataset_path,
        max_new_tokens=args.max_new_tokens,
        distributed=args.distributed,
        model_variant=args.model_variant,
        timing=Timing(args.timing),
        program_criteria_json_path=args.program_criteria_json_path,
        test_type=TestType(args.test_type),
        cross_entropy_threshold=args.cross_entropy_threshold,
        failure_rate_threshold=args.failure_rate_threshold,
        attention_type=AttnType(attention_type),
        prefill_chunk_size=args.prefill_chunk_size,
        stagger_load=args.stagger_load,
        stagger_update_lazyhandle=args.stagger_update_lazyhandle,
        dist_timeout=args.dist_timeout,
        run_cpu_validation=not args.skip_validation,
        validation_info_outputs_dir=args.validation_info_outputs_dir,
        save_validation_info_outputs=args.save_validation_info_outputs,
        prioritize_large_batch_sizes=args.prioritize_large_batch_sizes,
        enforce_homogeneous_prompt_programs=args.enforce_homogeneous_prompt_programs,
    )


if __name__ == "__main__":
    main()
