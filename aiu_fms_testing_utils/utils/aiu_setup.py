import os

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
        dprint(f"Sentient AIU: Enabled")
    else:
        dprint(f"Sentient AIU: Disabled (Senulator)")


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
        dprint(f"Detected running via torchrun")

    aiu_setup(rank, world_size)
