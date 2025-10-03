import os
import time
import logging
import argparse

import itertools
import torch

from fms.models import get_model
from fms.utils.generation import generate

from aiu_fms_testing_utils.testing.validation import get_default_validation_prefix

from aiu_fms_testing_utils.utils import prepare_inputs
from aiu_fms_testing_utils.utils.metrics_utils import tensor_abs_diff, tensor_cos_sim

from transformers import AutoTokenizer

logger = logging.getLogger(__name__)
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s %(message)s")

parser = argparse.ArgumentParser(
    description="Script to generate the model's metrics by layer"
)
parser.add_argument(
    "--architecture",
    type=str,
    help="The model architecture Eg.: hf_pretrained",
)
parser.add_argument(
    "--variant",
    type=str,
    default=None,
    help="The model variants (configuration) to benchmark. E.g. ibm-granite/granite-3.2-8b-instruct",
)
parser.add_argument(
    "--model_path",
    type=str,
    help="Paths to the directory containing model's weights (.pth files sharded by tensor parallel rank, not HF weights)",
)
parser.add_argument(
    "--mode",
    choices=["generate", "model-forward"],
    default="generate",
    required=True,
    help="Sets the output generation mode.",
)
parser.add_argument(
    "--batch_sizes",
    type=str,
    default="1",
    required=True,
    help="Batch sizes separated by comma. Eg.: 1,2",
)
parser.add_argument(
    "--seq_lengths",
    type=str,
    default="64",
    required=True,
    help="Sequence lengths separated by comma. Eg.: 64,2048",
)
parser.add_argument(
    "--max_new_tokens",
    type=str,
    default="128",
    required=True,
    help="Max number of generated tokens separated by comma. Eg.: 64,128",
)
parser.add_argument(
    "--output_path", type=str, default="/tmp/output", help="Path to save output files"
)
parser.add_argument(
    "--sharegpt_path",
    type=str,
    default=os.path.expanduser("~/share_gpt.json"),
    help="Path to sharegpt data json",
)

args = parser.parse_args()
mode = args.mode
output_path = args.output_path
sharegpt_path = args.sharegpt_path

common_model_paths = args.model_path if args.model_path else args.variant
if isinstance(common_model_paths, str):
    common_model_paths = [str(bs) for bs in common_model_paths.split(",")]

# pass custom common batch sizes as a comma separated str of ints
common_batch_sizes = args.batch_sizes
if isinstance(common_batch_sizes, str):
    common_batch_sizes = [int(bs) for bs in common_batch_sizes.split(",")]

# pass custom common seq lengths as a comma separated str of ints
common_seq_lengths = args.seq_lengths
if isinstance(common_seq_lengths, str):
    common_seq_lengths = [int(sl) for sl in common_seq_lengths.split(",")]

# pass custom common max new tokens as a comma separated str of ints
common_max_new_tokens = args.max_new_tokens
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

generate_iters = 0


def __infer_layer(model, max_len, device, max_new_tokens, batch_size, tokenizer):
    """
    Infer a model with registered layer hooks using generated inputs.

    Args:
        model (nn.Module): The model to infer.
        max_len (int): The maximum length of the input sequence.
        device (str): The device to use for inference.
        max_new_tokens (int): The maximum number of new tokens to generate.
        batch_size (int): The batch size for inference.
        tokenizer (Tokenizer): The tokenizer to use for encoding inputs.

    Returns:
        torch.Tensor: The inferred model's layers output.
    """

    do_sample = False
    use_cache = True

    prompts = prepare_inputs(batch_size, max_len, tokenizer, sharegpt_path)
    ids, pad_input_ids = prompts

    if "cuda" in device:
        ids = ids.to("cuda")

    if hasattr(model.config, "ntk_scaling") and model.config.ntk_scaling:
        max_seq_len = max(max_len, model.config.max_expected_seq_len)
    else:
        # without ntk scaling, extending the seq length too far gives bogus results.
        max_seq_len = model.config.max_expected_seq_len

    if "generate" in mode:
        with torch.no_grad():
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
            logger.info(f"Generation completed: Result len is {len(result)}")
            if len(result.shape) == 1:
                result = result.unsqueeze(0)
    else:
        result = model.forward(ids, use_cache=use_cache)
        logger.info(f"Model forward completed: Result len is {len(result)}")


def __register_call_layers(
    model, batch_size, device, seq_length, max_new_tokens, tokenizer
):
    """
    This function registers hooks on the model to track the forward pass of each layer.
    It returns a list of tuples containing the name and output of each layer in the model.

    Args:
        model (nn.Module): The PyTorch model to be analyzed.
        batch_size (int): The batch size used for inference.
        device (torch.device): The device on which the model is running.
        seq_length (int): The maximum sequence length of the input data.
        max_new_tokens (int): The maximum number of new tokens to be generated during inference.
        tokenizer (Tokenizer): The tokenizer used for tokenization.

    Returns:
        list: A list of tuples containing the name and output of each layer in the model.
    """
    layer_stack = {}
    pt_compile_model_time = time.time()

    module_depth = {}
    module_name = {}

    def register_depths(module, current_depth=0, name="model"):
        module_depth[module] = current_depth
        module_name[module] = name
        parent = name
        # if we are dealing with array of layers
        array_layers = all(key.isdigit() for key in module._modules.keys())
        for name, child in module._modules.items():
            if array_layers:
                register_depths(child, current_depth + 1, parent + "[" + name + "]")
            else:
                register_depths(child, current_depth + 1, parent + "." + name)

    register_depths(model)

    def wrap_forward(layer):
        original_forward = layer.forward

        def safe_forward(*args, **kwargs):
            try:
                return original_forward(*args, **kwargs)
            except (RuntimeError, TypeError) as e:
                logger.error(f"Error in {layer.__class__.__name__}: {e}")
                return torch.zeros_like(args[0]) if args else None

        layer.forward = safe_forward

    hooks = []

    def pre_hook_fn(module, input):
        depth = module_depth.get(module, 0)
        layer_name = module_name.get(module, 0)
        prefix = "│    " * depth
        if len(input) == 0:
            return
        input_shape_str = f"[{', '.join(map(str, input[0].shape))}]"
        input_type = str(input[0].dtype)
        if module.parameters() is None:
            return
        param_size = sum(p.numel() for p in module.parameters() if p.requires_grad)
        param_size_str = f"{param_size:,}" if param_size > 0 else "--"
        logger.info(
            f"{prefix}├─{layer_name}() -> {module.__class__.__name__} : | Input(arg): {input_shape_str} | {input_type} | Params: {param_size_str}"
        )
        wrap_forward(module)
        # save input for later use with outputs
        module._debug_input = input

    def post_hook_fn(module, input, output):
        layer_name = module_name.get(module, 0)
        # Save inputs and outputs
        tmp = {}
        if hasattr(module, "_debug_input"):
            global generate_iters
            generate_iters += 1
            layer_name = (
                f"{layer_name}.iter-{generate_iters}"
                if layer_name in layer_stack.keys()
                else layer_name
            )
            tmp[layer_name] = output
            layer_stack.update(tmp)
            # Clean up
            delattr(module, "_debug_input")

    for name, layer in model.named_modules():
        hooks.append(layer.register_forward_pre_hook(pre_hook_fn))
        hooks.append(layer.register_forward_hook(post_hook_fn))

    __infer_layer(
        model=model,
        max_len=seq_length,
        device=device,
        max_new_tokens=max_new_tokens,
        batch_size=batch_size,
        tokenizer=tokenizer,
    )

    for hook in hooks:
        hook.remove()

    pt_compile_model_time = time.time() - pt_compile_model_time
    logger.info(f"PT compile complete, took {pt_compile_model_time:.3f}s")

    return layer_stack


def get_metric_values(metric_list):
    if isinstance(metric_list, list):
        # shape of the first tensor to be printed in file
        metric_shape = metric_list[0].shape
        metric_list_res = torch.stack(metric_list).flatten().tolist()
    else:
        metric_shape = metric_list.shape
        metric_list_res = metric_list.flatten().tolist()

    return metric_list_res, metric_shape


def write_csv(values, path, metric, gpu_layer_shape, cpu_layer_shape, output_shape):
    """
    Write values to a CSV file at the given path.

    Args:
        values (list or float): A list of values to be written to the CSV file.
        If `values` is a single float, it will be written as a scalar value in the first column of the CSV file.
        path (str): The path to the CSV file to write to.
        metric (str): The name of the metric being evaluated.
        gpu_layer_shape (tuple): The shape of the GPU layer used for training.
        cpu_layer_shape (tuple): The shape of the CPU layer used for training.
        output_shape (tuple): The shape of the output generated by the model.

    Returns:
        None
    """
    with open(path, "w") as f:
        f.write(f"{metric}\n")
        f.write(f"GPU shape {gpu_layer_shape} CPU shape {cpu_layer_shape}\n")
        f.write(f"Metric shape {output_shape}\n")
        if not isinstance(values, float):
            for t in values:
                f.write(f"{t}\n")
        else:
            f.write(f"{values}\n")
        f.close()


def generate_layers_metrics(model_path, batch_size, seq_length, max_new_tokens):
    """
    Generate metrics for layers in a given model.

    Args:
        model_path (str): The path to the Hugging Face model.
        batch_size (int): The batch size used for inference.
        seq_length (int): The sequence length used for inference.
        max_new_tokens (int): The maximum number of new tokens allowed for generation.

    Returns:
        None
    """

    torch.manual_seed(42)
    os.environ["COMPILATION_MODE"] = "offline_decoder"

    if "HF_HOME" not in os.environ:
        os.environ["HF_HOME"] = "/tmp/models/hf_cache"

    model_path_kwargs = (
        {"variant": model_path} if args.variant else {"model_path": model_path}
    )
    micro_model_kwargs = {"architecture": args.architecture}

    get_model_kwargs = {
        **model_path_kwargs,
        **micro_model_kwargs,
    }

    tokenizer = AutoTokenizer.from_pretrained(model_path)

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

    layer_stack_cpu = __register_call_layers(
        model=validation_model,
        batch_size=batch_size,
        device="cpu",
        seq_length=seq_length,
        max_new_tokens=max_new_tokens,
        tokenizer=tokenizer,
    )

    global generate_iters
    generate_iters = 0
    logger.info("Finished registering CPU layers")

    layer_stack_cuda = __register_call_layers(
        model=validation_model_cuda,
        batch_size=batch_size,
        device="cuda",
        seq_length=seq_length,
        max_new_tokens=max_new_tokens,
        tokenizer=tokenizer,
    )

    assert len(layer_stack_cuda.keys()) == len(layer_stack_cpu.keys())

    for layer_key, output_val in layer_stack_cuda.items():
        tensor_cpu_out = None
        tensor_cuda_out = None

        if layer_key in layer_stack_cpu.keys():
            cpu_output = layer_stack_cpu[layer_key]
            cuda_output = output_val
            logger.info(f"Comparing CPU and GPU Layer {layer_key} output")

            if type(cpu_output) is tuple and type(cuda_output) is tuple:
                cos_sim = []
                abs_diff = []
                if len(cpu_output) < 2 and len(cpu_output[-1]) == 1:
                    # Projection layers (often called "query," "key," and "value" projections) are used to transform the input embeddings
                    # into separate query, key, and value vectors. They have tuple outputs, with more than 2 tensors - this path compares this type of output;
                    # In case of head layers, the last item of the tuple is a list of tensors with the same lenght as the
                    # number of layers in the model. The else path compares this other case.
                    tensor_cuda_out = cuda_output[-1]
                    tensor_cpu_out = cpu_output[-1]
                    for i in range(len(cpu_output)):
                        logger.debug(
                            f"inputs: {cuda_output[i].shape} {cpu_output[i].to('cuda').shape}"
                        )
                        cos_sim.append(
                            tensor_cos_sim(cuda_output[i], cpu_output[i].to("cuda"))
                        )
                        logger.debug(
                            f"cos_sim output:{tensor_cos_sim(cuda_output[i], cpu_output[i].to('cuda')).shape}"
                        )
                        abs_diff.append(
                            tensor_abs_diff(cuda_output[i], cpu_output[i].to("cuda"))
                        )
                else:
                    head_tensor_cpu = cpu_output[-1]
                    head_tensor_gpu = cuda_output[-1]
                    for i in range(len(head_tensor_gpu)):
                        if isinstance(head_tensor_gpu[i], tuple):
                            for j in range(len(head_tensor_gpu[i])):
                                tensor_cuda_out = head_tensor_gpu[i][j]
                                tensor_cpu_out = head_tensor_cpu[i][j]
                                logger.debug(
                                    f"inputs: {head_tensor_gpu[i][j].shape} {head_tensor_cpu[i][j].to('cuda').shape}"
                                )
                                cos_sim.append(
                                    tensor_cos_sim(
                                        head_tensor_cpu[i][j].to("cuda"),
                                        head_tensor_gpu[i][j],
                                    )
                                )
                                logger.debug(
                                    f"cos_sim output:{tensor_cos_sim(head_tensor_cpu[i][j].to('cuda'), head_tensor_gpu[i][j]).shape}"
                                )
                                abs_diff.append(
                                    tensor_abs_diff(
                                        head_tensor_cpu[i][j].to("cuda"),
                                        head_tensor_gpu[i][j],
                                    )
                                )
                        else:
                            tensor_cuda_out = head_tensor_gpu[i]
                            tensor_cpu_out = head_tensor_cpu[i]
                            logger.debug(
                                f"inputs: {head_tensor_gpu[i].shape} {head_tensor_cpu[i].to('cuda').shape}"
                            )
                            cos_sim.append(
                                tensor_cos_sim(
                                    head_tensor_cpu[i].to("cuda"), head_tensor_gpu[i]
                                )
                            )
                            logger.debug(
                                f"cos_sim output:{tensor_cos_sim(head_tensor_cpu[i].to('cuda'), head_tensor_gpu[i]).shape}"
                            )
                            abs_diff.append(
                                tensor_abs_diff(
                                    head_tensor_cpu[i].to("cuda"), head_tensor_gpu[i]
                                )
                            )
            else:
                tensor_cpu_out = cpu_output.to("cuda")
                tensor_cuda_out = cuda_output
                abs_diff = tensor_abs_diff(tensor_cpu_out, cuda_output)
                cos_sim = tensor_cos_sim(tensor_cpu_out, cuda_output)

            prefix = get_default_validation_prefix(
                model_id=model_path,
                max_new_tokens=max_new_token,
                batch_size=batch_size,
                seq_length=seq_length,
                dtype="float16",
            )
            layer_name = str(layer_key).replace("[", "").replace("]", "")

            abs_diff_path = os.path.join(
                output_path, f"{prefix}--{layer_name}.abs_diff.csv"
            )
            cos_sim_path = os.path.join(
                output_path, f"{prefix}--{layer_name}.cos_sim.csv"
            )

            cos_sim_res, cos_shape = get_metric_values(cos_sim)
            abs_diff_res, abs_diff_shape = get_metric_values(abs_diff)

            if not os.path.exists(abs_diff_path):
                logger.debug("saving abs_diff files")
                write_csv(
                    abs_diff_res,
                    abs_diff_path,
                    "abs_diff",
                    tensor_cuda_out.shape,
                    tensor_cpu_out.shape,
                    abs_diff_shape,
                )
            if not os.path.exists(cos_sim_path):
                logger.debug("saving cos_sim files")
                write_csv(
                    cos_sim_res,
                    cos_sim_path,
                    "cos_sim",
                    tensor_cuda_out.shape,
                    tensor_cpu_out.shape,
                    cos_shape,
                )

    logger.info(f"Completed {model_path} layers' metrics generation with {mode} mode")


for model_id, batch_size, sequence_length, max_new_token in common_shapes:
    logger.info(
        f"testing model_id-{model_id}, max_new_tokens-{max_new_token}, batch_size-{batch_size}, seq_length-{sequence_length}"
    )
    generate_layers_metrics(
        model_path=model_id,
        batch_size=batch_size,
        seq_length=sequence_length,
        max_new_tokens=max_new_token,
    )
