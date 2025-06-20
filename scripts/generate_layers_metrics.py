import os
import time

import itertools
import torch
import torch.nn as nn

from fms.utils import tokenizers
from fms.models import get_model
from fms.utils.generation import pad_input_ids, generate

from aiu_fms_testing_utils.testing.validation import get_default_validation_prefix

from aiu_fms_testing_utils.utils import (
    sample_sharegpt_requests,
    ids_for_prompt,
)
from aiu_fms_testing_utils.utils.aiu_setup import dprint



ORIGINAL_HF_HOME = os.environ.get("HF_HOME", None)

SHARE_GPT_DATASET_PATH = os.environ.get(
    "SHARE_GPT_DATASET_PATH", os.path.expanduser("~/share_gpt.json")
)

common_model_paths = common_model_paths = os.environ.get(
    "MODEL_PATHS",
    ["ibm-granite/granite-3.2-8b-instruct"],
)
common_batch_sizes = os.environ.get("BATCH_SIZES", [1, 2, 4, 8])
common_seq_lengths = os.environ.get("SEQ_LENGTHS", [64, 2048])
common_max_new_tokens = os.environ.get("MAX_NEW_TOKENS", [128])

output_dir = os.environ.get("OUTPUT_PATH", "/tmp/output")

# pass custom model path list for eg: EXPORT FMS_TESTING_COMMON_MODEL_PATHS="/tmp/models/granite-3-8b-base,/tmp/models/granite-7b-base"
if isinstance(common_model_paths, str):
    common_model_paths = common_model_paths.split(",")

# pass custom common batch sizes as a comma separated str of ints
if isinstance(common_batch_sizes, str):
    common_batch_sizes = [int(bs) for bs in common_batch_sizes.split(",")]

# pass custom common seq lengths as a comma separated str of ints
if isinstance(common_seq_lengths, str):
    common_seq_lengths = [int(sl) for sl in common_seq_lengths.split(",")]

# pass custom common max new tokens as a comma separated str of ints
if isinstance(common_max_new_tokens, str):
    common_max_new_tokens = [int(mnt) for mnt in common_max_new_tokens.split(",")]

common_shapes = list(
    itertools.product(
        common_model_paths,
        common_batch_sizes,
        common_seq_lengths,
        common_max_new_tokens,
    )
)

def __prepare_inputs(batch_size, seq_length, tokenizer, seed=0):
    prompts_and_sizes = sample_sharegpt_requests(
        SHARE_GPT_DATASET_PATH,
        batch_size,
        tokenizer,
        int(seq_length / 2),
        seq_length,
        seed,
    )
    print(prompts_and_sizes)
    ## TODO: for each prompt 
    prompt_list = []
    for prompt, _ in prompts_and_sizes:
        prompt_list.append(ids_for_prompt(prompt, tokenizer))

    input_ids, padding_kwargs = pad_input_ids(prompt_list, min_pad_length=seq_length)
    return input_ids, padding_kwargs

def __infer_layer(warmup, model, max_len, device,
                max_new_tokens, batch_size, tokenizer):
    

    do_sample = False
    use_cache = True

    prompts = __prepare_inputs(batch_size, max_len, tokenizer)
    ids, pad_input_ids = prompts

    if "cuda" in device:
        print("cuda prompts")
        print(len(prompts))
        ids = ids.to("cuda")
    
    if hasattr(model.config, "ntk_scaling") and model.config.ntk_scaling:
        max_seq_len = max(max_len, model.config.max_expected_seq_len)
    else:
        # without ntk scaling, extending the seq length too far gives bogus results.
        max_seq_len = model.config.max_expected_seq_len

    result = generate(
        model,
        ids,
        max_new_tokens=max_new_tokens,
        use_cache=use_cache,
        do_sample=do_sample,
        max_seq_len=max_seq_len,
        timing="e2e",
        eos_token_id=None,
        contiguous_cache=True,
        extra_kwargs={},
    )
    result, timings = result
    dprint(f"E2E timing information: {timings[0]:.3f}s")
    if len(result.shape) == 1:
        result = result.unsqueeze(0)

    if not warmup:
        for i in range(result.shape[0]):
            print(result[i])

def __register_call_layers(model, batch_size, device, seq_length, max_new_tokens, tokenizer):
    layer_stack = []
    pt_compile_model_time = time.time()

    module_depth = {}
    module_name = {}

    def register_depths(module, current_depth=0, name='model'):
        module_depth[module] = current_depth
        module_name[module] = name
        parent=name
        # if we are dealing with array of layers
        array_layers = all(key.isdigit() for key in module._modules.keys())
        for name, child in module._modules.items():
            if array_layers: 
                register_depths(child, current_depth + 1, parent+'['+name+']')
            else:
                register_depths(child, current_depth + 1, parent+'.'+name)

    register_depths(model)

    def wrap_forward(layer):
        original_forward = layer.forward

        def safe_forward(*args, **kwargs):
            #print("In safe forward")
            try:
                #print("No Error")
                return original_forward(*args, **kwargs)
            except (RuntimeError,TypeError) as e:
                print(f"Error in {layer.__class__.__name__}: {e}")
                return torch.zeros_like(args[0]) if args else None
        layer.forward = safe_forward
        

    hooks = []
    def pre_hook_fn(module, input):
        depth = module_depth.get(module, 0)
        layer_name = module_name.get(module, 0)
        prefix = '│    ' * depth
        if len(input) == 0: return
        input_shape_str = f"[{', '.join(map(str, input[0].shape))}]"
        input_type = str(input[0].dtype)
        if module.parameters() == None: return
        param_size = sum(p.numel() for p in module.parameters() if p.requires_grad)
        param_size_str = f"{param_size:,}" if param_size > 0 else "--"
        print(f"DEBUG TOOL {prefix}├─{layer_name}() -> {module.__class__.__name__} : | Input(arg): {input_shape_str} | {input_type} | Params: {param_size_str}")
        wrap_forward(module)
        # save input for later use with outputs
        module._debug_input = input 

    def post_hook_fn(module, input, output):
        print("post_hook_fn")
        layer_name = module_name.get(module, 0)
        # Save inputs and outputs
        if hasattr(module, '_debug_input'):
            print(output)
            layer_stack.append((layer_name, output))
            # Clean up
            delattr(module, '_debug_input')
    
    for name, layer in model.named_modules():
        hooks.append(layer.register_forward_pre_hook(pre_hook_fn))
        hooks.append(layer.register_forward_hook(post_hook_fn))

    
    __infer_layer(warmup=True, 
                  model= model, max_len=seq_length, 
                  device=device, max_new_tokens=max_new_tokens, 
                  batch_size=batch_size, tokenizer=tokenizer)

    for hook in hooks:
        hook.remove()

    pt_compile_model_time = time.time() - pt_compile_model_time
    dprint(f"PT compile complete, took {pt_compile_model_time:.3f}s")

    return layer_stack

def write_csv(l, path, metric):
    print("saving file")
    with open(path, 'w') as f:
        f.write(f'{metric}\n')
        if not type(l) is float:
            print("saving list")
            for t in l:
                f.write(f"{t}\n")
        else:
            print("saving float")
            f.write(f"{l}\n")
        f.close()

def convert_tensor(output):
    out_unique = set(list(itertools.chain.from_iterable(output)))
    keys = {key: value for key, value in zip(out_unique, range(len(out_unique)))}
    return torch.zeros(size=(len(output), len(keys)))

def generate_layers_metrics(model_path, batch_size, seq_length, max_new_tokens):
    torch.manual_seed(42)
    os.environ["COMPILATION_MODE"] = "offline_decoder"

    if "HF_HOME" not in os.environ:
        os.environ["HF_HOME"] = "/tmp/models/hf_cache"

    model_path_kwargs = {"variant": model_path}
    micro_model_kwargs = {"architecture": "hf_pretrained"}

    get_model_kwargs = {
        **model_path_kwargs,
        **micro_model_kwargs,
    }

    tokenizer = tokenizers.get_tokenizer(model_path)

    # prepare the cpu model
    validation_model = get_model(
        device_type="cpu",
        data_type=torch.float32,
        fused_weights=False,
        **get_model_kwargs,
    )

    # prepare the cuda model
    validation_model_cuda = get_model(
        device_type="cuda",
        data_type=torch.float16,
        fused_weights=False,
        **get_model_kwargs,
    )

    layer_stack_cpu = __register_call_layers(model=validation_model,
                                            batch_size=batch_size, 
                                            device="cpu", 
                                            seq_length=seq_length, max_new_tokens=max_new_tokens, 
                                            tokenizer=tokenizer)
    
    layer_stack_cuda = __register_call_layers(model=validation_model_cuda,
                                             batch_size=batch_size, 
                                             device="cuda", 
                                             seq_length=seq_length, max_new_tokens=max_new_tokens, 
                                             tokenizer=tokenizer)
    
    absolute_differences = []

    assert len(layer_stack_cuda) == len(layer_stack_cpu)

    for layer, cuda_output in layer_stack_cuda:
        tensor_cuda_out = None
        tensor_cpu_out = None
        abs_diff = None
        for cpu_layer, cpu_output in layer_stack_cpu:
            if cpu_layer == layer:
                print("CPU Layer {} GPU Layer {}".format(cpu_layer, layer))

                if not type(cuda_output) is tuple:
                    tensor_cuda_out = cuda_output
                else:
                    tensor_cuda_out = convert_tensor(cuda_output)
                if type(cpu_output) is tuple:
                    tensor_cpu_out = convert_tensor(cpu_output)
                else:
                    tensor_cpu_out = cpu_output.to('cuda')
                print("tensor converted... get torch abs diff")
                abs_diff = torch.abs(tensor_cpu_out - tensor_cuda_out).flatten().tolist()
                print("abs_diff calculated")
                cos = nn.CosineSimilarity(dim=1)
                cos_sim = cos(tensor_cpu_out, tensor_cuda_out)
                print(cos_sim)

        absolute_differences.append(abs_diff)
        print("abs_diff list appended")
        print(len(absolute_differences))

        prefix = get_default_validation_prefix(model_id, max_new_token, batch_size, 0, 'float16')
        layer_name = str(layer).replace('[','').replace(']', '')

        print("saving files")
        write_csv(abs_diff, os.path.join(output_dir, f"{prefix}--{layer_name}.abs_diff.csv"), "abs_diff")
        write_csv(cos_sim, os.path.join(output_dir, f"{prefix}--{layer_name}.cos_sim.csv"), "cos_sim")
        
    print(f"Completed {model_id} layers' metrics generation")

for model_id, batch_size, sequence_length, max_new_token in common_shapes:
    print("testing ", "model_id-", model_id, ", max_new_tokens-", max_new_token, ", batch_size-",batch_size, ", seq_length-",sequence_length)
    generate_layers_metrics(model_path=model_id, batch_size=batch_size, seq_length=sequence_length, max_new_tokens=max_new_token)
