import numpy as np
import torch
import torch.nn as nn


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
