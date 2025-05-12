
# Model Test Utility Scripts


## Test Decoders

Script meant to test various decoder models. It is used to test the accuracy of the decoder models and compare them with the reference implementation. The goal of this script is to ensure that the decoder is functioning correctly and producing the expected output for different inputs. It has capability to perform shape testing at different levels.


### Environment variables

These environment variables are used to configure the behavior of the script and customize its execution based on the user's needs.

- `HF_HOME`: The location where Hugging Face models are stored.
    - Default value: `None`
- `COMPILATION_MODE`: The mode for compiling the model.
    - Default value: not set.
- `FMS_TEST_SHAPES_USE_MICRO_MODELS`: Whether to use micro models or not.
    - Default value: `"1"`
- `FMS_TEST_SHAPES_DISTRIBUTED`: Whether to use distributed execution or not.
    - Default value: `"0"`
- `SHARE_GPT_DATASET_PATH`: The path for the GPT dataset.
    - Default value: the expanded form of "~/share_gpt.json".
- `FMS_TEST_SHAPES_FORCE_VALIDATION_LEVEL_1`: Whether to force validation level 1 or not.
    - Default value: `"0"`
- `FMS_TEST_SHAPES_VALIDATION_INFO_DIR`: The directory for validation information.
    - Default value: `"/tmp/models/validation_info"`
- `FMS_TEST_SHAPES_COMMON_MODEL_PATHS`: The list of common model paths.
    - Default value: [LLAMA_3p1_8B_INSTRUCT, GRANITE_3p2_8B_INSTRUCT].
- `FMS_TEST_SHAPES_FAILURE_THRESHOLD`: The failure rate threshold for validation level 1.
    - Default value: `"0.01"`
- `FMS_TEST_SHAPES_METRICS_THRESHOLD`: The metrics threshold for validation level 1.
    - Default value: (2.0, (-1.0e-8, 1.0e-8)).
- `FMS_TEST_SHAPES_SAVE_VALIDATION_INFO_OUTPUTS`: Whether to save validation info outputs or not.
    - Default value: `"0"`
- `FMS_TEST_SHAPES_COMMON_BATCH_SIZES`: The list of common batch sizes.
    - Default value: [1, 2, 4, 8].
- `FMS_TEST_SHAPES_COMMON_SEQ_LENGTHS`: The list of common sequence lengths.
    - Default value: [64, 2048].
- `FMS_TEST_SHAPES_COMMON_MAX_NEW_TOKENS`: The list of common max new tokens.
    - Default value: [128].

