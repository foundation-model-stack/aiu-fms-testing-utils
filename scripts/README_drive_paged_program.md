The [drive_paged_programs.py](https://github.com/foundation-model-stack/aiu-fms-testing-utils/blob/main/scripts/drive_paged_programs.py) is designed to run and validate paged programs using a specified model variant. 

It supports different attention types, including `paged` and `paged_fp8`, with the default set to `paged`. The supported dataset types are `sharegpt` and `rag_factoid`, with the default set to `sharegpt`.

The script can take many arguments/flags according to your needs, but a few notable flags are provided below. To see the description of all the command-line arguments that the script can parse, run it with `--help`.
- `--distributed`: the script can run tests in a distributed environment, utilizing multiple instances of AIU, for faster execution.
- `--skip_validation`: set it to true to skip CPU validation, which will make the script much faster.
- `--save_validation_info_outputs`: set it to true to save cpu validation outputs for later consumption. The saved outputs will allow you to reuse CPU logits.
- `--validation_info_outputs_dir`: path to directory containing validation info outputs. The use of saved outputs will avoid re-compute and will significantly reduce script execution time.
- `--program_criteria_json_path` and `--dataset_path`: for both of these arguments, make sure that the provided directory path exists on your system.
- `--programs`: Specified programs to run. Format: <program_id>:<batch_constraint>,<seq_len_constraint>
  <program_id> can be one of an int, *, or ?. If an int, it will choose the exact program id. If *, it will choose all programs that match the batch_constraint and seq_len_constraint criteria. If ?, it will choose one program that matches the batch_constraint and seq_len_constraint criteria.
  <batch_constraint> can be one of int or conditional expression on the batch size. Int will default to >= expression. Otherwise we can support >, >=, <, <=, == with a val.
  <seq_len_constraint> can be one of int or conditional expression on the sequence length. Int will default to >= expression. Otherwise we can support >, >=, <, <=, == with a val.
- `--enforce_homogeneous_prompt_programs`: Ensures all sequences in a batch would hit the same prefill program (by default we only ensure the largest prompt hits a specific prefill program)

The following examples demonstrate the usage of the script. Replace the `<valid_path>` with your directory path.

You may run the script in a distributed environment with multiple instances of AIU in production. However, for testing purposes, running the script on a single AIU can be very useful. Before we look at the examples for a distributed environment, let us first look at an example command to run on a single instance AIU.

```bash
# Run with 4K context length on a single AIU instance
VLLM_DT_MAX_BATCH_SIZE=4 VLLM_DT_MAX_CONTEXT_LEN=4096 HF_HUB_CACHE=/home/senuser/models/huggingface_cache/hub DT_DEEPRT_VERBOSE=-1 DTLOG_LEVEL=error python3 drive_paged_programs.py --max_new_tokens=8 --model_variant=ibm-granite/granite-3.3-8b-instruct --program_criteria_json_path=/<valid_path>/dpp-4k-5.json --dataset_path=/<valid_path>/ShareGPT_V3_unfiltered_cleaned_split.json --dataset_type=sharegpt --test_type=tokens
```
The following is a snippet of an output that the above command should produce.

```
...
...
[ 0/ 1]: PT compile complete, took 79.305s
[ 0/ 1]: extracted prompts in 1.9285 seconds
[ 0/ 1]: *** testing program 0 ***
[ 0/ 1]: program id: 0, valid prompt: (1, 2880), input shape: torch.Size([1, 2880])
[ 0/ 1]: Prompt Length: 2847
[ 0/ 1]: For Program 0 in sentence 1:
[ 0/ 1]: Prompt:

You are to write a on road scenario configuration in pbtxt format (with other road agents like vehicles and pedestrians) that can be loaded into a self driving car simulation system to check if ego vehicle (called nurobot) can make correct decisions. The configuration contains other road agents, their behaviors, and conditions to check if the ego vehicle's behavior is expected.

Here are some general instructions
- Check the final configuration against the protobuf definition to make sure it is syntax correct
- Ego vehicle is a planner agent
- Other road agents are line follower agents
- Make sure the geo locations are accurate up to 8 digits after decimal
...
...

Below is the scenario specific description:

The ego vehicle is traveling from location (lat: 37.1233212321, lng: -122.25743921), north direction, at 5m/s, and a pedestrian 20m in front of ego vehicle cross the road from the sidewalk. There is another tailgater vehicle behind ego vehicle. We want to create a runtime check in ego vehicle config that the vehicle stopped in front of the pedestrian successfully, and not breaking too hard (<3m/s^2) to avoid collision from behind.
[ 0/ 1]: CPU tokens:
[203, 203, 2538, 322, 10284, 2760, 3488, 436]
[ 0/ 1]: AIU tokens:
[203, 306, 33964, 26755, 12251, 203, 203, 7608]
[ 0/ 1]: CPU output:


Write the pbtxt configuration for
[ 0/ 1]: AIU output:

// PBTXT CONFIG

scene
[ 0/ 1]: all tests passed
```

More examples are provided below for the distributed environment, which utilizes multiple instances of AIU for faster execution.

```bash
# Run with 4K context length
VLLM_DT_MAX_BATCH_SIZE=4 VLLM_DT_MAX_CONTEXT_LEN=4096 HF_HUB_CACHE=/home/senuser/models/huggingface_cache/hub DT_DEEPRT_VERBOSE=-1 DTLOG_LEVEL=error torchrun --nproc-per-node=4 /home/senuser/aiu-fms-testing-utils/scripts/drive_paged_programs.py --max_new_tokens=8 --model_variant=ibm-granite/granite-3.3-8b-instruct --program_criteria_json_path=/<valid_path>/dpp-4k.json --dataset_path=/<valid_path>/ShareGPT_V3_unfiltered_cleaned_split.json --test_type=tokens --distributed

# Run with 8K context length
VLLM_DT_MAX_BATCH_SIZE=16 VLLM_DT_MAX_CONTEXT_LEN=8192 HF_HUB_CACHE=/home/senuser/models/huggingface_cache/hub DT_DEEPRT_VERBOSE=-1 DTLOG_LEVEL=error torchrun --nproc-per-node=4 /home/senuser/aiu-fms-testing-utils/scripts/drive_paged_programs.py --max_new_tokens=8 --model_variant=ibm-granite/granite-3.3-8b-instruct --program_criteria_json_path=/<valid_path>/dpp-8k-16.json --dataset_path=<valid_path>/ShareGPT_V3_unfiltered_cleaned_split.json --dataset_type=sharegpt --test_type=tokens --distributed

# Run with 16K context length using the rag_factoid dataset type and a program with a specific batch size and prompt length
VLLM_DT_MAX_BATCH_SIZE=4 VLLM_DT_MAX_CONTEXT_LEN=16384 HF_HUB_CACHE=/home/senuser/models/huggingface_cache/hub DT_DEEPRT_VERBOSE=-1 DTLOG_LEVEL=error torchrun --nproc-per-node=4 /home/senuser/aiu-fms-testing-utils/scripts/drive_paged_programs.py --max_new_tokens=8 --model_variant=ibm-granite/granite-3.3-8b-instruct --program_criteria_json_path=<valid_path>/dpp-16k.json --dataset_path=/<valid_path>/long_context_factoid_post_process.jsonl --dataset_type=rag_factoid --test_type=tokens --distributed --programs 0:4,16256

# Run with a 32K context length using the rag_factoid dataset type and a program with any batch size and a specific prompt length
EN_PREFILL_OPT=1 VLLM_DT_MAX_BATCH_SIZE=4 VLLM_DT_MAX_CONTEXT_LEN=32768 HF_HUB_CACHE=/home/senuser/models/huggingface_cache/hub DT_DEEPRT_VERBOSE=-1 DTLOG_LEVEL=error torchrun --nproc-per-node=4 /home/senuser/aiu-fms-testing-utils/scripts/drive_paged_programs.py --max_new_tokens=8 --model_variant=ibm-granite/granite-3.3-8b-instruct --program_criteria_json_path=/<valid_path>/dpp-32k.json --dataset_path=/<valid_path>/long_context_factoid_post_process.jsonl --dataset_type=rag_factoid --test_type=tokens --distributed --programs 0:0,32640
```

Examples that showcase the use of `programs` argument:

- programs: \*:0\,\<8192

  Will match all programs with any batch size (all batch sizes >=0) and sequence lengths upto 8192.

  Example command:
  ```
  VLLM_DT_MAX_BATCH_SIZE=32 VLLM_DT_MAX_BATCH_TKV_LIMIT=131072 VLLM_DT_MAX_CONTEXT_LEN=32768 FLEX_HDMA_P2PSIZE=268435456 DEM_COMPILE_VERSION=1 torchrun --nproc-per-node=4 <path_to_this_repo>/scripts/drive_paged_programs.py --programs \*:0\,\<8192  --max_new_tokens=32 --model_variant=<path>/granite-3.3-8b-instruct-FP8 --program_criteria_json_path=<valid_path>/program_criteria.json --dataset_path=<valid_path>/ShareGPT_V3_unfiltered_cleaned_split.json --dataset_type=sharegpt --test_type=metrics --cross_entropy_threshold=2.4444521379470827 --failure_rate_threshold=0.6 --attention_type=paged_fp8 --validation_info_outputs_dir=<output_path>/tmp_validation_info_dir --distributed --prioritize_large_batch_sizes --enforce_homogeneous_prompt_programs
  ```

- programs 0:4,16256

  Since a program_id 0 is specified, any prompt that meets batch size crteria >=4, and seq length >=16256 and would result in this program would be selected.

  Example command:
  ```
  VLLM_DT_MAX_BATCH_SIZE=4 VLLM_DT_MAX_CONTEXT_LEN=16384 HF_HUB_CACHE=/home/senuser/models/huggingface_cache/hub DT_DEEPRT_VERBOSE=-1 DTLOG_LEVEL=error torchrun --nproc-per-node=4 /home/senuser/aiu-fms-testing-utils/scripts/drive_paged_programs.py --max_new_tokens=8 --model_variant=ibm-granite/granite-3.3-8b-instruct --program_criteria_json_path=<valid_path>/dpp-16k.json --dataset_path=/<valid_path>/long_context_factoid_post_process.jsonl --dataset_type=rag_factoid --test_type=tokens --distributed --programs 0:4,16256
  ```

- programs: \?:0\,\<8192

  Will match any one program with any batch size (all batch sizes >=0) and sequence lengths upto 8192.

  Example command:
  ```
  VLLM_DT_MAX_BATCH_SIZE=32 VLLM_DT_MAX_BATCH_TKV_LIMIT=131072 VLLM_DT_MAX_CONTEXT_LEN=32768 FLEX_HDMA_P2PSIZE=268435456 DEM_COMPILE_VERSION=1 torchrun --nproc-per-node=4 <path_to_this_repo>/scripts/drive_paged_programs.py --programs \*:0\,\<8192  --max_new_tokens=32 --model_variant=<path>/granite-3.3-8b-instruct-FP8 --program_criteria_json_path=<valid_path>/program_criteria.json --dataset_path=<valid_path>/ShareGPT_V3_unfiltered_cleaned_split.json --dataset_type=sharegpt --test_type=metrics --cross_entropy_threshold=2.4444521379470827 --failure_rate_threshold=0.6 --attention_type=paged_fp8 --validation_info_outputs_dir=<output_path>/tmp_validation_info_dir --distributed --prioritize_large_batch_sizes --enforce_homogeneous_prompt_programs
  ```
