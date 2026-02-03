import os
from dataclasses import dataclass

from aiu_fms_testing_utils.utils.aiu_setup import dprint


@dataclass
class DPPRunnerConfig:
    """Class to configure parameters that may vary with model architecture"""

    # populated during setup
    num_blocks: int | None = None
    tkv_limit: int | None = None

    def _get_int_env(self, key: str, default: int, context: str) -> int:
        """
        Read an integer environment variable or use a default.
        Always emits a debug message explaining the choice.
        """
        value = os.environ.get(key)
        if value is None:
            dprint(f"{context}. Using default {key}={default}")
            return default

        try:
            parsed = int(value)
        except ValueError as e:
            raise ValueError(
                f"{context}. Invalid value for environment variable {key}: "
                f"expected an integer, got '{value}'"
            ) from e

        dprint(f"{context}. Using {key} from environment: {parsed}")
        return parsed

    def _configure_granite_3_8b(self, use_distributed, world_size, prefill_chunk_size):
        """Configure environment for granite 3 8b architecture \
        We are setting defaults for env variables not provided. \
        Config class is set in wrapper setup_config function."""

        if use_distributed and world_size == 4:
            ##Only set defaults for TP=4
            context = (
                "Model granite-3.3-8b (or compatible) "
                "with tensor parallel size 4 detected"
            )
            self.tkv_limit = self._get_int_env(
                key="VLLM_DT_MAX_BATCH_TKV_LIMIT",
                default=524288,
                context=context,
            )

            # these values are to be consistent with vllm for granite 3.3 8b instruct
            blocks_override = 8192 if prefill_chunk_size > 0 else 2080

            self.num_blocks = self._get_int_env(
                key="AFTU_PAGED_KVCACHE_NUM_BLOCKS_HINT",
                default=blocks_override,
                context=context,
            )

    def setup_config(
        self, model_variant, use_distributed, world_size, prefill_chunk_size
    ):
        """Set up environment variables and default values if not specified"""

        ## configure per model architecture
        if (
            "granite-3.3-8b-instruct" in model_variant
            or "granite-4.0-8b" in model_variant
        ):
            self._configure_granite_3_8b(
                use_distributed, world_size, prefill_chunk_size
            )

        ## global defaults (fallback)
        ## TODO: IN future we may remove defaults for unknown configurations \
        ## and require users to set the environment variables
        ## num_blocks is set in generate if not set here
        if self.tkv_limit is None:
            self.tkv_limit = self._get_int_env(
                key="VLLM_DT_MAX_BATCH_TKV_LIMIT",
                default=524288,
                context="Unknown model configuration",
            )

    def env_updates(self) -> dict[str, str]:
        """Returns a key/value of environment variables needed for model compile"""
        if self.tkv_limit is None:
            raise RuntimeError(
                "ModelConfig.env_updates() called before setup_config(). "
                "Call setup_config(...) first."
            )

        return {"VLLM_DT_MAX_BATCH_TKV_LIMIT": str(self.tkv_limit)}
