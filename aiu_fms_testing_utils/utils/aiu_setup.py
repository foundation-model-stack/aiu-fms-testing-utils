import argparse
import os
import torch

# ==============================================================
# Common utilities
# ==============================================================
# -------------
# Discover the world size and my rank (envars set by torchrun)
# https://pytorch.org/docs/stable/elastic/run.html#environment-variables
# -------------
local_rank = int(os.getenv("LOCAL_RANK", 0))
rank = int(os.getenv("RANK", 0))
world_rank = rank
world_size = int(os.getenv("WORLD_SIZE", 1))


def dprint_str(text):
    return f"[{rank:2d}/{world_size:2d}]: {text}"


def dprint(text):
    print(dprint_str(text))


# ==============================================================
# Common setup
# ==============================================================
def aiu_setup(rank=0, world_size=1, local_rank=0, local_size=1, verbose=False):
    # -------------
    # Envar setup for Sentient backend
    # -------------

    # Default to senulator backend unless user specified otherwise
    os.environ.setdefault("FLEX_COMPUTE", "SENULATOR")

    # FYI: This needs to be setup externally.
    # Since we are not setting 'DEEPRT_EXPORT_DIR' in the environment we cannot
    # query for it here. We are using the default value from deeptools.
    # if os.getenv("DUMP_MEMMAP") is not None:
    #     if os.getenv("SDSC_REF_DIR") is None:
    #         os.environ["SDSC_REF_DIR"] = os.environ["DEEPRT_EXPORT_DIR"]
    #     else:
    #         os.environ["SDSC_REF_DIR"] += f"/{rank}"
    #     assert (
    #         os.getenv("DUMP_MEMMAP_DIR") is not None
    #     ), "DUMP_MEMMAP_DIR not set while DUMP_MEMMAP set"
    #     os.environ["DUMP_MEMMAP_DIR"] += f"/{rank}"
    #     os.makedirs(
    #         os.environ["DUMP_MEMMAP_DIR"], exist_ok=True
    #     )  # directory needs to exist

    if os.getenv("FLEX_COMPUTE") == "SENTIENT":
        dprint("Sentient AIU: Enabled")
    else:
        dprint("Sentient AIU: Disabled (Senulator)")


# ==============================================================
# Distributed setup
# ==============================================================
def aiu_dist_setup(rank, world_size, local_rank=-0, local_size=-1, verbose=False):
    if local_rank < 0:
        local_rank = rank
    if local_size < 0:
        local_size = world_size

    if os.getenv("TORCHELASTIC_RUN_ID") is None:
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12355"
    elif rank == 0 or verbose:
        dprint("Detected running via torchrun")

    aiu_setup(rank, world_size)


# ==============================================================
# Environment variables utilities
# ==============================================================
def set_aiu_env_vars(args: argparse.Namespace) -> None:
    """Set necessary environment variables for AIU

    NOTE: to enable graph export, set DTCOMPILER_KEEP_EXPORT=true in your env
    """

    if not args.is_encoder:
        if not args.compile_dynamic:
            _target_cache_size = max(
                int(args.max_new_tokens * 2),
                int(args.min_pad_length * 2.5),
                int(args.fixed_prompt_length * 2.5),
            )
            _prompt_size = max(int(args.min_pad_length), int(args.fixed_prompt_length))
            if hasattr(torch._dynamo.config, "accumulated_cache_size_limit"):
                if (
                    _target_cache_size
                    > torch._dynamo.config.accumulated_cache_size_limit
                ):
                    _prev = torch._dynamo.config.accumulated_cache_size_limit
                    torch._dynamo.config.accumulated_cache_size_limit = (
                        _target_cache_size
                    )
                    dprint(
                        "NOTICE: Adjusting torch._dynamo.config.accumulated_cache_size_limit "
                        f"from {_prev} to {torch._dynamo.config.accumulated_cache_size_limit} "
                        f"to accomodate prompt size of {_prompt_size} and decode tokens of "
                        f"{args.max_new_tokens}"
                    )
            if _target_cache_size > torch._dynamo.config.cache_size_limit:
                _prev = torch._dynamo.config.cache_size_limit
                torch._dynamo.config.cache_size_limit = _target_cache_size
                dprint(
                    f"NOTICE: Adjusting torch._dynamo.config.cache_size_limit from {_prev} to "
                    f"{torch._dynamo.config.cache_size_limit} to accomodate prompt size of "
                    f"{_prompt_size} and decode tokens of {args.max_new_tokens}"
                )
            torch._dynamo.config.assume_static_by_default = True
            torch._dynamo.config.automatic_dynamic_shapes = False

        os.environ.setdefault("COMPILATION_MODE", "offline_decoder")

    if args.device_type == "aiu-senulator":
        os.environ["FLEX_COMPUTE"] = "SENULATOR"
        os.environ["FLEX_DEVICE"] = "MOCK"
    else:
        if "AIU_WORLD_RANK_0" not in os.environ:
            print("must set AIU_WORLD_RANK_0")
            exit()
        os.environ.setdefault("FLEX_COMPUTE", "SENTIENT")
        os.environ.setdefault("FLEX_DEVICE", "PF")  # will use VF eventually
