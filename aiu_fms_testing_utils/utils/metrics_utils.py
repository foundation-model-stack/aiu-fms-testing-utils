import numpy as np
import torch
import torch.nn as nn


def get_model_prefix(model_path, 
                     shapes_size, 
                     max_new_tokens: None, 
                     batch_size: None, 
                     seq_length: None, 
                     dtype: None, 
                     include_shapes: False):
    """
    Generate a prefix for a model based on its path and other parameters.

    Args:
        model_path (str): The path to the model file.
        shapes_size (int): The size of the shapes array to use in the model.
        max_new_tokens (int): The maximum number of new tokens to use in the model.
        batch_size (int): The batch size to use in the model.
        seq_length (int): The sequence length to use in the model.
        dtype (str): The data type to use in the model.
        include_shapes (bool): Include or not the shapes to the prefix.
    Returns:
        str: A prefix for the model based on its path and other parameters.
    """
    if model_path.count("/") > 1:
        # this means that the model_path does NOT match to the hf pattern 
        # Eg.: /home/another-dir/another/ibm-granite/granite-3.3-8b-base
        model_prefix = model_path.split("/")[-2] + "--" + model_path.split("/")[-1]
    else:
        # this means that the model_path does match to the hf pattern 
        # Eg.: ibm-granite/granite-3.3-8b-base
        model_prefix = model_path.replace("/", "--")

    if shapes_size > 1 or include_shapes:
        model_prefix = f"{model_prefix}_max-new-tokens-{max_new_tokens}_batch-size-{batch_size}_seq-length-{seq_length}_dtype-{dtype}"
        
    return model_prefix

def abs_diff_linalg_norm(res_vector):
    """
    Calculates the Euclidean norm (also known as the L2 norm) of a given array res_vector. This is equivalent to finding the square
    root of the sum of the squares of all the elements in the array. It's a fundamental operation in linear algebra and is often used 
    to measure the "length" or "magnitude" of a vector. More at https://numpy.org/devdocs/reference/generated/numpy.linalg.norm.html
    Args:
        res_vector (list): The list of abs diff

    Returns:
        float: "magnitude" of the diff vector.
    """
    return np.linalg.norm(res_vector)

def list_mean(val_list):
    """
    Calculates the mean for all the values in a given list.
    Args:
        val_list (list): The list of values

    Returns:
        float: mean value calculated.
    """
    return np.mean(val_list)

def tensor_abs_diff(tensor1, tensor2):
    """
    Calculate the absolute difference between two tensors.

    Args:
        tensor1 (torch.Tensor): The first input tensor.
        tensor2 (torch.Tensor): The second input tensor.

    Returns:
        torch.Tensor: The absolute difference tensor.

    Example:
        >>> tensor1 = torch.tensor([1, 2, 3])
        >>> tensor2 = torch.tensor([4, 5, 6])
        >>> abs_diff(tensor1, tensor2)
        torch.tensor([3, 3, 3])
    """
    abs_diff = torch.abs(tensor1 - tensor2)
    return abs_diff
                    
def tensor_cos_sim(tensor1, tensor2):
    """
    Computes the cosine similarity between two tensors.

    Args:
        tensor1 (torch.Tensor): The first input tensor.
        tensor2 (torch.Tensor): The second input tensor.

    Returns:
        torch.Tensor: The cosine similarity between the two input tensors.

    Example:
        >>> import torch
        >>> tensor1 = torch.randn(3, 5)
        >>> tensor2 = torch.randn(3, 5)
        >>> sim = cos_sim(tensor1, tensor2)
        >>> print(sim)
    """
    cos = nn.CosineSimilarity(dim=-1)
    tensor1[tensor1 == 0.0] = 1e-6
    tensor2[tensor2 == 0.0] = 1e-6
    cos_sim = cos(tensor1, tensor2)
    return cos_sim