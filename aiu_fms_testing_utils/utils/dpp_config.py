import os

from aiu_fms_testing_utils.utils.aiu_setup import dprint

class ModelConfig:
    """Class to configure parameters that may vary with model architecture"""
    
    num_blocks: int
    tkv_limit: int

    def configure_granite_3_8b(self, use_distributed, world_size, prefill_chunk_size):
        """Configure environment for granite 3 8b architecture \
        We are setting defaults for env variables not provided. \
        Config class is set in wrapper setup_config function"""
        if (
            use_distributed
            and world_size() == 4
        ):
            ## Only set defaults for TP=4
            if not os.getenv("VLLM_DT_MAX_BATCH_TKV_LIMIT"):           
                os.environ["VLLM_DT_MAX_BATCH_TKV_LIMIT"] = 524288        
                dprint(
                    f"Model granite-3.3-8b-instruct and tensor parallel size 4 detected. \
                    Using default value of VLLM_DT_MAX_BATCH_TKV_LIMIT for this known \
                    configuration."
                )

            # these values are to be consistent with vllm for granite 3.3 8b instruct
            if prefill_chunk_size > 0:
                blocks_override = 8192
            else:
                blocks_override = 2080

            if not os.getenv("AFTU_PAGED_KVCACHE_NUM_BLOCKS_HINT"):   
                dprint(
                    f"Model granite-3.3-8b-instruct and tensor parallel size 4 detected. \
                    Using default value of AFTU_PAGED_KVCACHE_NUM_BLOCKS_HINT for this known \
                    configuration."
                )
                os.environ["AFTU_PAGED_KVCACHE_NUM_BLOCKS_HINT"] = blocks_override


    def setup_config(self, model_variant, use_distributed, world_size, prefill_chunk_size):
        """Set up environment variables and default values if not specified"""


        ## configure per model architecture
        if "granite-3.3-8b-instruct" in model_variant:
            self.configure_granite_3_8b(use_distributed, world_size, prefill_chunk_size)
        
        ## set default config and do this code once here \
        ## after we looped over all model architectures 
        ## TODO: IN future we may remove defaults for unknown architectures \
        ## and require users to set the environment variables
        if not os.getenv("AFTU_PAGED_KVCACHE_NUM_BLOCKS_HINT"):
            dprint(
                f"Unknown configuration found. \
                Setting DPP default for AFTU_PAGED_KVCACHE_NUM_BLOCKS_HINT. For best results, \
                specify the environment variable and re-run."
            )
            os.environ["AFTU_PAGED_KVCACHE_NUM_BLOCKS_HINT"] = 8192     

        self.num_blocks_override = int(os.environ.get("AFTU_PAGED_KVCACHE_NUM_BLOCKS_HINT"))

        if not os.getenv("VLLM_DT_MAX_BATCH_TKV_LIMIT"):
            dprint(
                f"Unknown configuration found. \
                Setting DPP default for VLLM_DT_MAX_BATCH_TKV_LIMIT. For best results, \
                specify the environment variable and re-run."
            )
            os.environ["VLLM_DT_MAX_BATCH_TKV_LIMIT"] = 524288

        self.tkv_limit = int(os.environ.get("VLLM_DT_MAX_BATCH_TKV_LIMIT"))


            
            
            