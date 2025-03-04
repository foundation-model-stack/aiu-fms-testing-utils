import torch
import torch.nn as nn
import time
from fms.utils.generation import generate
from aiu_fms_testing_utils.utils.aiu_setup import dprint

def warmup_model(model: nn.Module, input_ids: torch.Tensor, max_new_tokens: int, **padding_kwargs):
    from torch_sendnn import torch_sendnn
    dprint("AIU warmup")
    pt_compile_model_time = time.time()
    extra_kwargs = {**padding_kwargs, "only_last_token": True}
    generate(model, input_ids, max_new_tokens=max_new_tokens, max_seq_len=model.config.max_expected_seq_len, use_cache=True, do_sample=False, extra_kwargs=extra_kwargs)
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