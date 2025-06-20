# Standard
from pathlib import Path
import argparse
import itertools
import os
import sys
import time

# Third Party
from fms.utils import generation
from fms.utils.generation import generate, pad_input_ids
from fms.utils.tokenizers import BaseTokenizer
from torch import nn
import numpy as np
import torch

# Local Packages
from aiu_fms_testing_utils.utils import warmup_model
from aiu_fms_testing_utils.utils.aiu_setup import dprint, local_rank


class DecoderInfer():
    """Run inference (generation) with LLM decoder models."""

    def __init__(
        self,
        model: nn.Module,
        tokenizer: BaseTokenizer,
        args: argparse.Namespace,
        device: torch.device,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.args = args
        self.device = device

        self.add_special_tokens = False
        self.has_padding = True
        self.max_len = 0
        self.extra_generation_kwargs = {}

        # !!! Inference arguments (hardcoded, as in the original script)
        self.do_sample = [False]
        self.use_cache = [args.no_use_cache]  # True/False identical with greedy iff `torch.use_deterministic_algorithms(True)`

        self.validate_decoder_arguments()

    def validate_decoder_arguments(self):
        """Ensure arguments compatibility with Encoder models."""

        args = self.args
        if getattr(args, "is_encoder", True):
            raise ValueError(
                "Running decoder model but is_encoder argument is either not set or True"
            )
        if "bert" in args.architecture.lower():
            raise ValueError(
                f"Architecture {args.architecture} should be run as an encoder model."
            )

    def ids_for_prompt(self, prompt):
        """Process textual prompt and return tokenized ids."""

        tokens = self.tokenizer.tokenize(prompt)
        ids = self.tokenizer.convert_tokens_to_ids(tokens)
        if self.add_special_tokens:
            ids = [self.tokenizer.bos_token_id] + ids
        ids = torch.tensor(ids, dtype=torch.long, device=self.device)
        return ids

    def truncate_prompts_to_max_length(self, prompts, max_len, max_allowed_length):
        """Truncate a series of prompts to a selected max length.
        This function ensures prompt truncation prior to padding the input ids."""

        if max_allowed_length is not None and max_len > max_allowed_length:
            dprint(f"max prompt length is {max_len}, truncate to {max_allowed_length}")
            prompts = [prompt[:max_allowed_length] for prompt in prompts]
        return prompts

    def process_eval_set(self):
        """Load textual prompts from file or use defaults prompts, then convert them
        to ids.
        """

        args = self.args
        self.add_special_tokens = (
            self.tokenizer.bos_token_id != self.tokenizer.eos_token_id
        )

        if args.prompt_path != "":
            # Before creating the Path object, check if prompt_path has a glob pattern
            if isinstance(args.prompt_path, str):
                prompt_path, sep, glob_pattern = args.prompt_path.partition("*")
            else:
                sep = ""
                glob_pattern = ""
            glob_pattern = sep + glob_pattern

            prompt_path = Path(os.path.expanduser(prompt_path))
            prompt_file_paths = []

            if prompt_path.is_dir():
                if glob_pattern != "":
                    glob_pattern_list = [glob_pattern]
                else:
                    glob_pattern_list = ["*.txt"]
                for glob_pattern_possibility in glob_pattern_list:
                    file_list = list(prompt_path.glob(glob_pattern_possibility))
                    if len(file_list) > 0:
                        prompt_file_paths = sorted(file_list)
                        break

            if prompt_path.is_file():
                prompt_file_paths = [prompt_path]

            # Check if we found some files
            assert len(prompt_file_paths) > 0, f"Can't find any prompt files at {prompt_path}"

            # Check if we have enough files
            assert (
                len(prompt_file_paths) >= args.batch_size
            ), f"Not enough prompt files at {prompt_path} for a batch size of {args.batch_size}"

            prompts = []
            for i, prompt_file_path in enumerate(prompt_file_paths):
                if i == args.batch_size:
                    break
                prompts.append(self.ids_for_prompt(prompt_file_path.read_text(encoding="utf-8")))
        else:
            if args.prompt_type == "chat":
                template = "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{}\n\n### Response:"

                prompt1 = template.format(
                    "Provide a list of instructions for preparing chicken soup."
                )
                prompt2 = template.format("Explain some popular greetings in Spanish.")
                prompt3 = template.format("Explain to me why ignorance is bliss.")
                prompt4 = template.format(
                    "I have just come into a very large sum of money. Provide me a list of things that I can do with my new found wealth."
                )
            elif args.prompt_type == "code":
                template = "[INST] Write code to solve the following coding problem that obeys the constraints and passes the example test cases. Please wrap your code answer using ```:\n{}\n[/INST]"
                prompt1 = template.format("Write a bubble sort function in python.")
                prompt2 = template.format(
                    "Using the Java streams API, write a simple function which will get the cumulative sum of a list of integers."
                )
                prompt3 = template.format(
                    "In bash, how do I list all directories and sub-directories which contain a .py file."
                )
                prompt4 = template.format(
                    "Write a simple decorator in python which will modify all string inputs to ints if possible."
                )
            else:
                dprint("prompt_type must be one of chat or code")
                exit()

            prompt1 = self.ids_for_prompt(prompt1)
            prompt2 = self.ids_for_prompt(prompt2)
            prompt3 = self.ids_for_prompt(prompt3)
            prompt4 = self.ids_for_prompt(prompt4)
            prompts = [prompt1, prompt2, prompt3, prompt4]
            prompts = prompts * ((args.batch_size // 4) + 1)
            prompts = prompts[: args.batch_size]

        if args.fixed_prompt_length != 0:
            padding_length = args.fixed_prompt_length
            max_allowed_length = args.fixed_prompt_length
        else:
            padding_length = args.min_pad_length
            max_allowed_length = args.max_prompt_length

        self.has_padding = args.batch_size > 1 or padding_length != 0
        self.max_len = max([len(prompt) for prompt in prompts])

        if args.fixed_prompt_length != 0 and args.fixed_prompt_length < self.max_len:
            dprint(
                "One or more prompts require truncation. Truncation has been disabled "
                "because fixed_prompt_length was set."
            )
            sys.exit(1)
        prompts = self.truncate_prompts_to_max_length(
            prompts,
            self.max_len,
            max_allowed_length,
        )
        if self.has_padding:
            ids, extra_generation_kwargs = pad_input_ids(
                prompts,
                min_pad_length=padding_length,
            )
        else:
            ids = prompts
            if isinstance(ids, list) and len(ids) == 1:
                ids = ids[0].unsqueeze(0)
            extra_generation_kwargs = {}

        self.extra_generation_kwargs = extra_generation_kwargs

        return ids

    def print_result(self, result, result_idx: int):
        """Printout generation output."""

        args = self.args

        if local_rank != 0:
            return
        if self.has_padding:
            result = generation.trim_prefix(result)

        result = generation.trim_prefix(result, self.tokenizer.bos_token_id)

        # stop at EOS token if present and remove padding
        if not args.no_early_termination:
            result = generation.truncate_after_eos(result, self.tokenizer.eos_token_id)

        output_str = self.tokenizer.convert_tokens_to_string(
            self.tokenizer.convert_ids_to_tokens(result)
        )

        if args.output_path != "":
            output_path = Path(args.output_path)
            output_path.mkdir(parents=True, exist_ok=True)
            if output_path.is_dir():
                file_path = output_path / f"{result_idx}.txt"
                with file_path.open("w", encoding="utf-8") as file:
                    file.write(output_str + "\n")
        dprint(output_str)
        print()

    def infer(self, ids, warmup):
        """Run generation inference (warmup compiled model or per-warmed generation).

        NOTE: with greedy generation (do_sample=False) we _should_ always get the same
        results. However, there is currently a bug in start_pos for batched rotary
        embeddings that can lead varying results for the same prompt.
        """

        args = self.args

        for sample, cache in itertools.product(self.do_sample, self.use_cache):
            if local_rank == 0 and not warmup:
                dprint(f"use_cache {cache} | do_sample {sample}")
                dprint("==================")
            if (
                hasattr(self.model.config, "ntk_scaling")
                and self.model.config.ntk_scaling
            ):
                max_seq_len = max(self.max_len, self.model.config.max_expected_seq_len)
            else:
                # w/o ntk scaling, extending the seq length too far gives bogus results
                max_seq_len = self.model.config.max_expected_seq_len

            # Add only_last_token optimization
            self.extra_generation_kwargs["only_last_token"] = True

            if args.device_type == "cpu":
                self.extra_generation_kwargs["attn_algorithm"] = "math"

            if not args.no_early_termination and not warmup:
                eos_token_id = self.tokenizer.eos_token_id
            else:
                eos_token_id = None

            result = generate(
                self.model,
                ids,
                max_new_tokens=args.max_new_tokens,
                use_cache=cache,
                do_sample=sample,
                max_seq_len=max_seq_len,
                timing=args.timing,
                eos_token_id=eos_token_id,
                contiguous_cache=True,
                extra_kwargs=self.extra_generation_kwargs,
            )
            if args.timing != "":
                result, timings = result
                if args.timing == "e2e":
                    dprint(f"E2E timing information: {timings[0]:.3f}s")
                elif args.timing == "per-token":
                    if not warmup:
                        dprint(f"First-token latency: {timings[0]*1000:.3f} ms")
                        dprint(f"Average next-token latency (including first token): {np.mean(timings)*1000:.3f} ms")
                        if len(timings) > 1:
                            dprint(f"Average next-token latency: {np.mean(timings[1:])*1000:.3f} ms")
                            dprint(f"Max next-token latency: {np.max(timings[1:])*1000:.3f} ms (token #{np.argmax(timings[1:]) + 2})")
                            dprint(f"Min next-token latency: {np.min(timings[1:])*1000:.3f} ms (token #{np.argmin(timings[1:]) + 2})")
                            dprint(f"Std deviation of next-token latencies: {np.std(timings[1:])*1000:.3f} ms")
                    timings = [f"{t*1000:.3f}" for t in timings]
                    dprint(f"Per-token timing information: {', '.join(timings)} ms")
            if len(result.shape) == 1:
                result = result.unsqueeze(0)

            if not warmup:
                for i in range(result.shape[0]):
                    self.print_result(result[i], i)

    def run_warmup(self, ids):
        """Run warmup cycle of compiled model."""

        dprint(f"Start compilation warmup...")
        pt_compile_model_start = time.time()
        if self.args.device_type == "aiu":  # only run warmup for AIU, not senulator
            warmup_model(
                self.model,
                ids,
                self.args.max_new_tokens,
                self.args.compile_dynamic_sendnn,
                **self.extra_generation_kwargs,
            )
            aiu_warmup_start = time.time()
            self.infer(ids, warmup=True)
            aiu_warmup_time = time.time() - aiu_warmup_start
            dprint(f"AIU warmup completed, took {aiu_warmup_time:.3f}s")
        else:
            for sample, cache in itertools.product(self.do_sample, self.use_cache):
                self.infer(cache, sample, True)
        pt_compile_model_time = time.time() - pt_compile_model_start
        dprint(f"PT compile complete, took {pt_compile_model_time:.3f}s")

    def run_generation(self, ids):
        """Run inference generation (not a warmup)."""

        dprint(f"Start generating output...")
        for _ in range(self.args.iters):
            self.infer(ids, warmup=False)


def run_decoder_eval(
        model: nn.Module,
        tokenizer: BaseTokenizer,
        args: argparse.Namespace,
        device: torch.device,
    ) -> None:
    """Entry point to run evaluation of LLM decoder models."""

    decoder_infer = DecoderInfer(model, tokenizer, args, device)
    ids = decoder_infer.process_eval_set()
    if args.compile:
        decoder_infer.run_warmup(ids)
    decoder_infer.run_generation(ids)
