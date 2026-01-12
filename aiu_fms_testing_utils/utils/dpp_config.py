from dataclasses import dataclass, field
import os

from aiu_fms_testing_utils.utils.aiu_setup import dprint

@dataclass
class ModelConfig:
    """Class to configure parameters that may vary with model architecture"""
    
    # populated during setup
    num_blocks: int | None = None
    tkv_limit: int|None = None

    def configure_granite_3_8b(self, use_distributed, world_size, prefill_chunk_size):
        """Configure environment for granite 3 8b architecture \
        We are setting defaults for env variables not provided. \
        Config class is set in wrapper setup_config function."""

        if use_distributed and world_size == 4:
            env_tkv = os.environ.get("VLLM_DT_MAX_BATCH_TKV_LIMIT")
            ## Only set defaults for TP=4
            if env_tkv is None:           
                self.tkv_limit = 524288    
                dprint(
                    f"Model granite-3.3-8b-instruct and tensor parallel size 4 detected."
                    f"Using default value of VLLM_DT_MAX_BATCH_TKV_LIMIT. {self.tkv_limit}"
                )
            else:
                self.tkv_limit = int(env_tkv)
                dprint(f"Using VLLM_DT_MAX_BATCH_TKV_LIMIT from environment: {self.tkv_limit}")

            # these values are to be consistent with vllm for granite 3.3 8b instruct
            blocks_override = 8192 if prefill_chunk_size > 0 else 2080
            env_blocks = os.environ.get("AFTU_PAGED_KVCACHE_NUM_BLOCKS_HINT")

            if env_blocks is None:   
                self.num_blocks = blocks_override
                dprint(
                    f"Model granite-3.3-8b-instruct and tensor parallel size 4 detected."
                    f"Using default value of AFTU_PAGED_KVCACHE_NUM_BLOCKS_HINT {self.num_blocks}"
                )              
            else:
                self.num_blocks = int(env_blocks)
                dprint(f"Using AFTU_PAGED_KVCACHE_NUM_BLOCKS_HINT from environment: {self.num_blocks}")


    def setup_config(self, model_variant, use_distributed, world_size, prefill_chunk_size):
        """Set up environment variables and default values if not specified"""

        ## configure per model architecture
        if "granite-3.3-8b-instruct" in model_variant:
            self.configure_granite_3_8b(use_distributed, world_size, prefill_chunk_size)
        
        ## global defaults (fallback)
        ## TODO: IN future we may remove defaults for unknown configurations \
        ## and require users to set the environment variables
        if self.num_blocks is None:
            env_blocks = os.environ.get("AFTU_PAGED_KVCACHE_NUM_BLOCKS_HINT")
            if env_blocks is None:
                dprint(
                    "Unknown configuration found."
                    "Using DPP default for AFTU_PAGED_KVCACHE_NUM_BLOCKS_HINT."
                    "For best results, specify the environment variable and re-run."
                )
                # finalize derived values
                self.num_blocks = 8192
            else:
                self.num_blocks = int(env_blocks)

        if self.tkv_limit is None:
            env_tkv = os.environ.get("VLLM_DT_MAX_BATCH_TKV_LIMIT")
            if env_tkv is None:
                dprint(
                    "Unknown configuration found."
                    "Using DPP default for VLLM_DT_MAX_BATCH_TKV_LIMIT."
                    "For best results, specify the environment variable and re-run."
                )
                self.tkv_limit = 524288
            else:
                self.tkv_limit = int(env_tkv)
                dprint(f"Using VLLM_DT_MAX_BATCH_TKV_LIMIT from environment: {self.tkv_limit}")


            
            
            