from typing import Any, Dict, NamedTuple, Tuple
from dataclasses import dataclass
from enum import Enum
import torch


class DeviceType(Enum):
    CPU = "cpu"
    SPYRE = "spyre"


class TestType(Enum):
    METRICS = "metrics"
    TOKENS = "tokens"


class AttnType(Enum):
    SDPA = "sdpa_causal"
    MATH = "math"
    PAGED = "spyre_paged_attn"
    MATH_FP8 = "math_fp8"
    PAGED_FP8 = "spyre_paged_attn_fp8"


@dataclass
class ProgramInfo:
    """Encapsulates program execution criteria.

    Attributes:
        program_id: Unique identifier for the program being tested.
        batch_size_limit: Numeric threshold for batch size constraint.
        batch_size_limit_type: Comparison operator for batch size (e.g., ">=", "<=", "==").
        prompt_length_limit: Numeric threshold for prompt length constraint.
        prompt_length_limit_type: Comparison operator for prompt length (e.g., ">=", "<=", "==")."""

    program_id: str
    batch_size_limit: int
    batch_size_limit_type: str
    prompt_length_limit: int
    prompt_length_limit_type: str


class EnvConfig(NamedTuple):
    """Represents global configuration derived from environment and CLI.

    Attributes:
        attn_name: The internal name of the attention algorithm.
        cpu_dtype: Data type for CPU validation ('fp8' or 'fp32').
        max_batch_size: Maximum batch size.
        max_tkv: Maximum total key-value (context) length."""

    attn_type: AttnType
    cpu_dtype: str
    max_batch_size: int
    max_tkv: int


class MetricResult(NamedTuple):
    """Result of comparing AIU and CPU logit distributions.

    Attributes:
        cross_entropy_loss: Cross-entropy loss between the distributions.
        mean_abs_diff: Mean absolute difference of softmax probabilities."""

    cross_entropy_loss: float
    mean_abs_diff: float

    def __str__(self) -> str:
        return f"cross_entropy_loss: {self.cross_entropy_loss:.6f}, mean_abs_diff: {self.mean_abs_diff:.6f}"


class PreparedInputs(NamedTuple):
    """Represents prepared model inputs from dataset sampling.

    Attributes:
        input_ids: Padded tensor of tokenized input IDs with shape (batch_size, seq_length).
        extra_kwargs: Dictionary with attention mask and other model inputs.
        sample_key: String identifier for the sampled prompts."""

    input_ids: torch.Tensor
    extra_kwargs: Dict[str, Any]
    sample_key: str


class ValidPrompt(NamedTuple):
    """Represents a valid prompt configuration for program execution.

    Attributes:
        program_id: ID of the program this prompt will execute.
        shape: Tuple of (batch_size, seq_length) for this prompt.
        input_ids: Tokenized and padded input tensor.
        extra_kwargs: Dictionary with attention mask and other model inputs.
        sample_key: String identifier for the sampled prompts."""

    program_id: str
    shape: Tuple[int, int]
    input_ids: torch.Tensor
    extra_kwargs: Dict[str, Any]
    sample_key: str
