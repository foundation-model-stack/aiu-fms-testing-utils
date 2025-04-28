import torch
import torch.nn as nn
import time
from fms.utils.tokenizers import BaseTokenizer
from fms.utils.generation import generate
from aiu_fms_testing_utils.utils.aiu_setup import dprint
from typing import Optional, List, Tuple
import os
import requests
import json
import random

def warmup_model(model: nn.Module, input_ids: torch.Tensor, max_new_tokens: int, **padding_kwargs):
    from torch_sendnn import torch_sendnn
    dprint("AIU warmup")
    pt_compile_model_time = time.time()
    extra_kwargs = {**padding_kwargs, "only_last_token": True}
    generate(model, input_ids, max_new_tokens=max_new_tokens, max_seq_len=model.config.max_expected_seq_len, use_cache=True, do_sample=False, contiguous_cache=True, extra_kwargs=extra_kwargs)
    pt_compile_model_time = time.time() - pt_compile_model_time
    dprint(f"PT compile complete, took {pt_compile_model_time:.3f}s")

    dprint("executing update_lazyhandle and performing validation")
    update_lh_time = time.time()
    torch_sendnn.update_lazyhandle()
    update_lh_time = time.time() - update_lh_time
    dprint(f"update_lazyhandle complete, took {update_lh_time:.3f}s")

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

        with open(filename, 'wb') as file:
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
    seed: Optional[int] = None
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
    seed: Optional[int] = None
) -> List[Tuple[str, int]]:
    if not os.path.exists(dataset_path):
        print("downloading share-gpt dataset as it does not exist")
        __download_file("https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json", dataset_path)

    # Load the dataset.
    with open(dataset_path, encoding='utf-8') as f:
        dataset = json.load(f)
    # Filter out the conversations with less than 2 turns.
    dataset = [data for data in dataset if len(data["conversations"]) >= 2]
    dataset = [data["conversations"][0]["value"] for data in dataset]

    return __sample_requests(dataset, num_requests, tokenizer, prompt_length_min, prompt_length_max, seed)

def sample_squad_v2_qa_requests(
    dataset_path: str,
    num_requests: int,
    tokenizer: BaseTokenizer,
    prompt_length_min: int = 32,
    prompt_length_max: int = 64,
    seed: Optional[int] = None
) -> List[Tuple[str, int]]:
    from datasets import load_dataset

    if os.path.exists(dataset_path):
        ds = load_dataset(dataset_path)['train']
    else:
        ds = load_dataset("rajpurkar/squad_v2", cache_dir=dataset_path)['train']


    ds = [f"{data['context']}\n{data['question']}" for data in ds]

    return __sample_requests(ds, num_requests, tokenizer, prompt_length_min, prompt_length_max, seed)


