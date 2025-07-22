# Standard
from typing import Optional, List, Tuple
import json
import os
import random
import requests
import time

# Third Party
from aiu_fms_testing_utils.utils.aiu_setup import dprint
from fms.utils.tokenizers import BaseTokenizer
from fms.utils.generation import pad_input_ids
import torch
import torch.nn as nn


def warmup_model(
    model: nn.Module,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    compile_dynamic_sendnn: bool = False,
    use_cache: bool = True,
    **extra_kwargs,
):
    import torch_sendnn

    attention_specific_kwargs = {}
    attn_name = extra_kwargs["attn_name"]
    if "paged" in attn_name:
        from aiu_fms_testing_utils.utils.paged import generate, adjust_inputs_to_batch
    else:
        # TODO: Add a unified generation dependent on attn_type
        from fms.utils.generation import generate

        attention_specific_kwargs["contiguous_cache"] = True
        attention_specific_kwargs["max_seq_len"] = input_ids.shape[1] + max_new_tokens

    dprint("AIU warmup")
    pt_compile_model_time = time.time()

    # adjust inputs depending on attn_type and dynamic shapes
    _warmup_input_ids = input_ids
    _extra_kwargs = extra_kwargs
    _max_new_tokens = max_new_tokens
    if compile_dynamic_sendnn:
        _max_new_tokens = 2
        # always warmup with batch size 2 when using attn_type=paged
        if "paged" in attn_name:
            _warmup_input_ids, _extra_kwargs = adjust_inputs_to_batch(
                input_ids,
                **extra_kwargs,
            )

    extra_kwargs = {**_extra_kwargs, "only_last_token": "paged" not in attn_name}

    with torch_sendnn.warmup_mode():
        generate(
            model,
            _warmup_input_ids,
            max_new_tokens=_max_new_tokens,
            do_sample=False,
            use_cache=use_cache,
            extra_kwargs=extra_kwargs,
            **attention_specific_kwargs,
        )
    pt_compile_model_time = time.time() - pt_compile_model_time
    dprint(f"PT compile complete, took {pt_compile_model_time:.3f}s")


def ids_for_prompt(prompt, tokenizer):
    tokens = tokenizer.tokenize(prompt)
    ids = tokenizer.convert_tokens_to_ids(tokens)
    if tokenizer.bos_token_id != tokenizer.eos_token_id:
        ids = [tokenizer.bos_token_id] + ids
    ids = torch.tensor(ids, dtype=torch.long, device="cpu")
    return ids


def __download_file(url, filename):
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()

        with open(filename, "wb") as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
        print(f"Successfully downloaded {filename}")

    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")


def __sample_requests(
    prompt_list: List[str],
    num_requests: int,
    tokenizer: BaseTokenizer,
    prompt_length_min: int = 32,
    prompt_length_max: int = 64,
    seed: Optional[int] = None,
):
    # Shuffle the dataset.
    if seed is not None:
        random.Random(seed).shuffle(prompt_list)

    # Filter out sequences that are too long or too short
    filtered_dataset: List[Tuple[str, int, int]] = []
    for i in range(len(prompt_list)):
        if len(filtered_dataset) == num_requests:
            break

        # Tokenize the prompts and completions.
        prompt = prompt_list[i]
        prompt_token_ids = ids_for_prompt(prompt, tokenizer)

        prompt_len = len(prompt_token_ids)
        if prompt_len < prompt_length_min or prompt_len > prompt_length_max:
            # Prune too short or too long sequences.
            continue
        filtered_dataset.append((prompt, prompt_len))

    return filtered_dataset


def sample_sharegpt_requests(
    dataset_path: str,
    num_requests: int,
    tokenizer: BaseTokenizer,
    prompt_length_min: int = 32,
    prompt_length_max: int = 64,
    seed: Optional[int] = None,
) -> List[Tuple[str, int]]:
    if not os.path.exists(dataset_path):
        print("downloading share-gpt dataset as it does not exist")
        __download_file(
            "https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json",
            dataset_path,
        )

    # Load the dataset.
    with open(dataset_path, encoding="utf-8") as f:
        dataset = json.load(f)
    # Filter out the conversations with less than 2 turns.
    dataset = [data for data in dataset if len(data["conversations"]) >= 2]
    dataset = [data["conversations"][0]["value"] for data in dataset]

    return __sample_requests(
        dataset,
        num_requests,
        tokenizer,
        prompt_length_min,
        prompt_length_max,
        seed,
    )


def sample_squad_v2_qa_requests(
    dataset_path: str,
    num_requests: int,
    tokenizer: BaseTokenizer,
    prompt_length_min: int = 32,
    prompt_length_max: int = 64,
    seed: Optional[int] = None,
) -> List[Tuple[str, int]]:
    from datasets import load_dataset

    if os.path.exists(dataset_path):
        ds = load_dataset(dataset_path)["train"]
    else:
        ds = load_dataset("rajpurkar/squad_v2", cache_dir=dataset_path)["train"]

    ds = [f"{data['context']}\n{data['question']}" for data in ds]

    return __sample_requests(
        ds,
        num_requests,
        tokenizer,
        prompt_length_min,
        prompt_length_max,
        seed,
    )


def prepare_inputs(
    batch_size, seq_length, tokenizer, ds_path, seed=0, ds_type="sharegpt"
):
    """
    Prepare input IDs and padding kwargs for a batch of questions.

    Args:
        batch_size (int): The number of questions in the batch.
        seq_length (int): The maximum length of the input sequence.
        tokenizer (Tokenizer): A tokenizer object to tokenize the questions.
        ds_path (str): The path to the dataset file.
        seed (int, optional): The random seed for reproducibility. Defaults to 0.
        ds_type (str, optional): The type of dataset to use. Can be "sharegpt" or any other supported dataset type. Defaults to "sharegpt".

    Returns:
        tuple: A tuple containing the input IDs and padding kwargs.
    """
    if "sharegpt" not in ds_type:
        prompts_and_sizes = sample_squad_v2_qa_requests(
            ds_path,
            batch_size,
            tokenizer,
            int(seq_length / 2),
            seq_length,
            seed,
        )
    else:
        prompts_and_sizes = sample_sharegpt_requests(
            ds_path,
            batch_size,
            tokenizer,
            int(seq_length / 2),
            seq_length,
            seed,
        )
    prompt_list = []
    for prompt, _ in prompts_and_sizes:
        prompt_list.append(ids_for_prompt(prompt, tokenizer))

    input_ids, padding_kwargs = pad_input_ids(prompt_list, min_pad_length=seq_length)
    return input_ids, padding_kwargs
