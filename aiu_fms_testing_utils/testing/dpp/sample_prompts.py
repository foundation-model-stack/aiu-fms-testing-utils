from aiu_fms_testing_utils.testing.utils import format_kwargs_to_string
from aiu_fms_testing_utils.utils import (
    get_pad_size,
    sample_rag_factoid_requests,
    sample_sharegpt_requests,
)
from aiu_fms_testing_utils.utils.aiu_setup import dprint, local_rank


from transformers import AutoTokenizer


from pathlib import Path


def get_sampler(dataset_type: str, dataset_path: str, tokenizer: AutoTokenizer):
    """Selects and configures the sampler based on type.

    Returns a sampler function and configuration for the specified dataset type.

    Args:
        dataset_type: Type of dataset ("custom", "rag_factoid", or "sharegpt").
        dataset_path: Path to the dataset file or directory.
        tokenizer: HuggingFace tokenizer for encoding prompts.

    Returns:
        Tuple containing:
            - sampler: Callable function for sampling prompts from the dataset.
            - allow_truncation: Boolean indicating if prompt truncation is allowed.
            - custom_shape: Tuple of (batch_size, max_seq_length) for custom datasets,
                           None for other dataset types.

    Raises:
        ValueError: If dataset_type is not one of the supported types.
        SystemExit: If custom dataset path is not a directory or file reading fails.
    """
    custom_shape = None
    if dataset_type == "custom":
        if local_rank == 0:
            dprint(
                "Using custom prompts from user, programs parameter will be ignored as it will be determined by user prompt"
            )
        directory = Path(dataset_path)
        if not directory.is_dir():
            dprint("when using a custom dataset, you must provide a directory")
            exit()

        result = []
        for fp in directory.iterdir():
            if fp.is_file():
                try:
                    content = fp.read_text()
                    result.append(
                        (content, get_pad_size(len(tokenizer.encode(content))))
                    )
                except Exception as e:
                    print(f"Error while reading {fp} for custom dataset: {e}")
                    exit()

        custom_shape = (len(result), max([_[1] for _ in result]))

        def _custom_line_sampler(**kwargs):
            """Custom sampler for user-provided text files.

            Returns pre-loaded prompts from custom dataset files without
            additional sampling logic. Supports optional sample key return.

            Args:
                **kwargs: Keyword arguments, supports "return_key" flag.

            Returns:
                List of (prompt, padded_size) tuples, or tuple of (list, sample_key)
                if return_key=True.
            """
            return_key = kwargs.get("return_key", False)
            sample_key = format_kwargs_to_string(**kwargs)
            if return_key:
                return result, sample_key
            return result

        sampler = _custom_line_sampler
        allow_truncation = False
    elif dataset_type == "rag_factoid":
        sampler = sample_rag_factoid_requests
        allow_truncation = False
    elif dataset_type == "sharegpt":
        sampler = sample_sharegpt_requests
        allow_truncation = True
    else:
        raise ValueError("dataset_type must be one of rag_factoid or sharegpt")

    return sampler, allow_truncation, custom_shape
