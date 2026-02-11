import argparse

from aiu_fms_testing_utils.testing.dpp.run_drive_paged_programs import run_dpp


def main() -> None:
    """Main execution function for driving paged program validation tests.

    Workflow:
    1. Sets and configures environment.
    2. Loads models (both AIU-compiled and CPU validation).
    3. Warms up the model.
    4. Selects programs and prompts to test based on criteria.
    5. For each program/prompt combination:
       - Generates CPU validation data (or loads from cache).
       - Runs AIU inference.
       - Compares outputs using metrics or token-based validation.
    6. Prints results and failure cases.

    Raises:
        Various exceptions may be raised during:
        - Model loading (e.g., OOM, invalid model variant)
        - Distributed initialization (e.g., timeout, network issues)
        - File I/O (e.g., missing program criteria JSON)
        - Validation (e.g., shape mismatches)
    """

    parser = argparse.ArgumentParser(
        description="Script which will drive paged programs for debugging"
    )
    parser.add_argument(
        "--program_criteria_json_path",
        type=str,
        help="path to json file containing the program criteria list",
        required=True,
    )
    args = parser.parse_args()

    programs = "2:0,<8192"
    max_new_tokens = 32
    distributed = True
    model_variant = "ibm-granite/granite-3.3-8b-instruct"
    timing = ""
    program_criteria_json_path = args.program_criteria_json_path
    dataset_path = "sharegpt"
    test_type = "tokens"
    cross_entropy_threshold = 2.6
    failure_rate_threshold = 0.1
    attention_type = "paged"
    prefill_chunk_size = 1024
    stagger_load = 0
    stagger_update_lazyhandle = 0
    dist_timeout = 0
    skip_validation = True
    validation_info_outputs_dir = "/home/senuser/models/validation_info"
    save_validation_info_outputs = False
    prioritize_large_batch_sizes = True
    enforce_homogeneous_prompt_programs = True

    attention_map = {
        "sdpa": "sdpa_causal",
        "paged": "spyre_paged_attn",
        "math_fp8": "math_fp8",
        "paged_fp8": "spyre_paged_attn_fp8",
    }

    run_dpp(
        programs=programs,
        dataset_path=dataset_path,
        max_new_tokens=max_new_tokens,
        distributed=distributed,
        model_variant=model_variant,
        timing=timing,
        program_criteria_json_path=program_criteria_json_path,
        test_type=test_type,
        cross_entropy_threshold=cross_entropy_threshold,
        failure_rate_threshold=failure_rate_threshold,
        attention_type=attention_map[attention_type],
        prefill_chunk_size=prefill_chunk_size,
        stagger_load=stagger_load,
        stagger_update_lazyhandle=stagger_update_lazyhandle,
        dist_timeout=dist_timeout,
        run_cpu_validation=not skip_validation,
        validation_info_outputs_dir=validation_info_outputs_dir,
        save_validation_info_outputs=save_validation_info_outputs,
        prioritize_large_batch_sizes=prioritize_large_batch_sizes,
        enforce_homogeneous_prompt_programs=enforce_homogeneous_prompt_programs,
    )


if __name__ == "__main__":
    main()
