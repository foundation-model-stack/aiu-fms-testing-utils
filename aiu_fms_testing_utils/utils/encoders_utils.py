# Standard
from tqdm import tqdm
import argparse
import collections
import json
import os
import time

# Third Party
from datasets import Dataset, load_dataset
from fms.models.hf import to_hf_api
from fms.models.hf.modeling_hf_adapter import HFModelArchitecture
from fms.utils import has_package
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from torch import nn
from torch.utils.data import DataLoader
import evaluate
import numpy as np
import torch

# Local Packages
from aiu_fms_testing_utils.utils.aiu_setup import dprint, rank


# Optional imports (required for QA)
has_hf = has_package("transformers")
if has_hf:
    from transformers import (
        default_data_collator,
        DataCollatorWithPadding,
        EvalPrediction,
        pipeline,
    )


def wrap_encoder(model: nn.Module) -> HFModelArchitecture:
    """Add config info and wrapper to run pipeline for RoBERTa MaskedLM."""

    if not has_hf:
        raise ImportError(
            "MaskedLM Encoder requires transformers package but import "
            "was unsuccessful."
        )

    model.config.linear_config.pop("linear_type", None)
    return to_hf_api(model, task_specific_params=None)


def move_to_device(batch: dict, device: torch.device) -> dict:
    """Move batch to selected device."""

    batch_on_device = {}
    for k, v in batch.items():
        batch_on_device[k] = v.to(device)
    return batch_on_device


class EncoderQAInfer:
    """Run QuestionAnswering task with encoder models."""

    def __init__(
        self,
        model: nn.Module,
        tokenizer: PreTrainedTokenizerBase,
        args: argparse.Namespace,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.args = args

        self.question_column_name = ""
        self.context_column_name = ""
        self.answer_column_name = ""
        self.pad_on_right = True

        self.validate_encoder_arguments()

    def validate_encoder_arguments(self) -> None:
        """Ensure arguments compatibility with Encoder models.

        NOTE: when Decoder models are refactored, this function will be expanded to
        ensure decoder arguments are not being provided to the encoder script.
        """

        args = self.args
        if not getattr(args, "is_encoder", False):
            raise ValueError(
                "Running encoder model but is_encoder argument is not set to True. "
                "Verify your launch script."
            )

    def prepare_validation_features(
        self,
        examples: dict[str, list[str | dict]],
    ) -> dict[str, list]:
        """Validation preprocessing"""

        args = self.args

        q_col_name = self.question_column_name
        c_col_name = self.context_column_name
        pad_on_right = self.pad_on_right
        max_prompt_length = (
            args.max_prompt_length
            if args.max_prompt_length is not None
            else 384  # this default is targeted at QA task (not a model limitation)
        )

        # Some of the questions have lots of whitespace on the left, which is not useful
        # and will make the truncation of the context fail (the tokenized question will
        # take a lots of space). So we remove that left whitespace
        examples[q_col_name] = [q.lstrip() for q in examples[q_col_name]]

        # Tokenize our examples with truncation and maybe padding, but keep the overflows
        # using a stride. This results in one example possible giving several features
        # when a context is long, each of those features having a context that overlaps
        # a bit the context of the previous feature.
        tokenized_examples = self.tokenizer(
            examples[q_col_name if pad_on_right else c_col_name],
            examples[c_col_name if pad_on_right else q_col_name],
            truncation="only_second" if pad_on_right else "only_first",
            max_length=max_prompt_length,
            stride=min(args.doc_stride, max_prompt_length // 2),
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length" if args.pad_to_max_length else False,
        )

        # Since one example might give us several features if it has a long context, we
        # need a map from a feature to its corresponding example.
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")

        # For evaluation, we will need to convert our predictions to substrings of the
        # context, so we keep the corresponding example_id and we will store the offset
        # mappings.
        tokenized_examples["example_id"] = []

        for i in range(len(tokenized_examples["input_ids"])):
            # Grab the sequence corresponding to that example (to know what is the
            # context and what is the question).
            sequence_ids = tokenized_examples.sequence_ids(i)
            context_index = 1 if pad_on_right else 0

            # One example can give several spans, this is the index of the example
            # containing this span of text.
            sample_index = sample_mapping[i]
            tokenized_examples["example_id"].append(examples["id"][sample_index])

            # Set to None the offset_mapping that are not part of the context so
            # it's easy to determine if a token position is part of the context or not.
            tokenized_examples["offset_mapping"][i] = [
                (o if sequence_ids[k] == context_index else None)
                for k, o in enumerate(tokenized_examples["offset_mapping"][i])
            ]

        return tokenized_examples

    def convert_batch_to_fms_style(
        self,
        batch: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """FMS uses a different standard than HF for encoder inputs.

        The mask is also handled differently in FMS: it is correctly processed by SDPA
        only if provided as boolean. A floating binary mask would not be converted.
        """

        return {"x": batch["input_ids"], "mask": batch["attention_mask"].to(torch.bool)}

    def process_eval_set(self) -> None:
        """Pre-process evaluation dataset for QuestionAnswering task."""

        if not has_hf:
            raise ImportError(
                "QuestionAnswering Encoder requires transformers package but import "
                "was unsuccessful."
            )

        args = self.args
        if args.dataset_name is not None:
            # Downloading and loading a dataset from the hub
            raw_datasets = load_dataset(
                args.dataset_name,
                args.dataset_config_name,
                trust_remote_code=False,
            )
        else:
            data_files = {}
            if args.validation_file is not None:
                data_files["validation"] = args.validation_file
                extension = args.validation_file.split(".")[-1]
            else:
                raise ValueError(
                    "Could not determine evaluation dataset to load. Pass `dataset_name` "
                    "or `validation_file` argument."
                )
            raw_datasets = load_dataset(extension, data_files=data_files, field="data")

        column_names = raw_datasets["train"].column_names

        self.question_column_name = (
            "question" if "question" in column_names else column_names[0]
        )
        self.context_column_name = (
            "context" if "context" in column_names else column_names[1]
        )
        self.answer_column_name = (
            "answers" if "answers" in column_names else column_names[2]
        )

        # Padding side determines if we do (question|context) or (context|question)
        self.pad_on_right = self.tokenizer.padding_side == "right"

        model_max_length = self.tokenizer.model_max_length
        if args.max_prompt_length > model_max_length:
            dprint(
                f"max_prompt_length ({args.max_prompt_length}) is larger than the "
                f"maximum length supported ({model_max_length}). "
                f"Using max_prompt_length={model_max_length} instead."
            )
            self.max_prompt_length = min(
                args.max_prompt_length,
                model_max_length,
            )

        eval_examples = raw_datasets["validation"]
        if args.max_eval_samples is not None:
            # We will select sample from whole data
            eval_examples = eval_examples.select(range(args.max_eval_samples))
        self.eval_examples = eval_examples

        eval_dataset = eval_examples.map(
            self.prepare_validation_features,
            batched=True,
            num_proc=args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not args.overwrite_cache,
            desc="Running tokenizer on validation dataset",
        )

        if args.max_eval_samples is not None:
            # During Feature creation dataset samples might increase, we will select
            # required samples again
            eval_dataset = eval_dataset.select(range(args.max_eval_samples))

        # store evaluation dataset prior dropping
        self.eval_dataset = eval_dataset

        # DataLoaders creation:
        if args.pad_to_max_length:
            # If padding was already done to max length, we use the default data collator
            # that will just convert everything to tensors.
            self.data_collator = default_data_collator
        else:
            # Otherwise, `DataCollatorWithPadding` will pad to the maximum length
            # of the samples passed.
            pad_to_multiple_of = None
            self.data_collator = DataCollatorWithPadding(
                self.tokenizer,
                pad_to_multiple_of=pad_to_multiple_of,
            )

        self.eval_dataset_for_model = eval_dataset.remove_columns(
            ["example_id", "offset_mapping"]
        )
        self.eval_dataloader = DataLoader(
            self.eval_dataset_for_model,
            shuffle=False,
            collate_fn=self.data_collator,
            batch_size=args.batch_size,
        )
        dprint("Dataloader initialized.")

        self.metric = evaluate.load(
            "squad_v2" if args.version_2_with_negative else "squad"
        )
        dprint("Evaluation metric initialized.")

    def postprocess_qa_predictions(
        self,
        examples: Dataset,
        features: Dataset,
        predictions: tuple[np.ndarray, np.ndarray],
        version_2_with_negative: bool = False,
        n_best_size: int = 20,
        max_answer_length: int = 30,
        null_score_diff_threshold: float = 0.0,
        output_dir: str | None = None,
        prefix: str | None = None,
    ) -> None:
        """
        Post-processes the predictions of a question-answering model to convert them to answers that are substrings of the
        original contexts. This is the base postprocessing functions for models that only return start and end logits.

        Args:
            examples: The non-preprocessed dataset (see the main script for more information).
            features: The processed dataset (see the main script for more information).
            predictions (:obj:`Tuple[np.ndarray, np.ndarray]`):
                The predictions of the model: two arrays containing the start logits and the end logits respectively. Its
                first dimension must match the number of elements of :obj:`features`.
            version_2_with_negative (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not the underlying dataset contains examples with no answers.
            n_best_size (:obj:`int`, `optional`, defaults to 20):
                The total number of n-best predictions to generate when looking for an answer.
            max_answer_length (:obj:`int`, `optional`, defaults to 30):
                The maximum length of an answer that can be generated. This is needed because the start and end predictions
                are not conditioned on one another.
            null_score_diff_threshold (:obj:`float`, `optional`, defaults to 0):
                The threshold used to select the null answer: if the best answer has a score that is less than the score of
                the null answer minus this threshold, the null answer is selected for this example (note that the score of
                the null answer for an example giving several features is the minimum of the scores for the null answer on
                each feature: all features must be aligned on the fact they `want` to predict a null answer).

                Only useful when :obj:`version_2_with_negative` is :obj:`True`.
            output_dir (:obj:`str`, `optional`):
                If provided, the dictionaries of predictions, n_best predictions (with their scores and logits) and, if
                :obj:`version_2_with_negative=True`, the dictionary of the scores differences between best and null
                answers, are saved in `output_dir`.
            prefix (:obj:`str`, `optional`):
                If provided, the dictionaries mentioned above are saved with `prefix` added to their names.
            log_level (:obj:`int`, `optional`, defaults to ``logging.WARNING``):
                ``logging`` log level (e.g., ``logging.WARNING``)
        """

        if len(predictions) != 2:
            raise ValueError(
                "`predictions` should be a tuple with two elements (start_logits, end_logits)."
            )
        all_start_logits, all_end_logits = predictions

        if len(predictions[0]) != len(features):
            raise ValueError(
                f"Got {len(predictions[0])} predictions and {len(features)} features."
            )

        # Build a map example to its corresponding features.
        example_id_to_index = {k: i for i, k in enumerate(examples["id"])}
        features_per_example = collections.defaultdict(list)
        for i, feature in enumerate(features):
            features_per_example[example_id_to_index[feature["example_id"]]].append(i)

        # The dictionaries we have to fill.
        all_predictions = collections.OrderedDict()
        all_nbest_json = collections.OrderedDict()
        if version_2_with_negative:
            scores_diff_json = collections.OrderedDict()

        # Logging.
        dprint(
            f"Post-processing {len(examples)} example predictions split into {len(features)} features."
        )

        # Let's loop over all the examples!
        for example_index, example in enumerate(tqdm(examples)):
            # Those are the indices of the features associated to the current example.
            feature_indices = features_per_example[example_index]

            min_null_prediction = None
            prelim_predictions = []

            # Looping through all the features associated to the current example.
            for feature_index in feature_indices:
                # We grab the predictions of the model for this feature.
                start_logits = all_start_logits[feature_index]
                end_logits = all_end_logits[feature_index]
                # This is what will allow us to map some the positions in our logits to span of texts in the original
                # context.
                offset_mapping = features[feature_index]["offset_mapping"]
                # Optional `token_is_max_context`, if provided we will remove answers that do not have the maximum context
                # available in the current feature.
                token_is_max_context = features[feature_index].get(
                    "token_is_max_context", None
                )

                # Update minimum null prediction.
                feature_null_score = start_logits[0] + end_logits[0]
                if (
                    min_null_prediction is None
                    or min_null_prediction["score"] > feature_null_score
                ):
                    min_null_prediction = {
                        "offsets": (0, 0),
                        "score": feature_null_score,
                        "start_logit": start_logits[0],
                        "end_logit": end_logits[0],
                    }

                # Go through all possibilities for the `n_best_size` greater start and end logits.
                start_indexes = np.argsort(start_logits)[
                    -1 : -n_best_size - 1 : -1
                ].tolist()
                end_indexes = np.argsort(end_logits)[
                    -1 : -n_best_size - 1 : -1
                ].tolist()
                for start_index in start_indexes:
                    for end_index in end_indexes:
                        # Don't consider out-of-scope answers, either because the indices are out of bounds or correspond
                        # to part of the input_ids that are not in the context.
                        if (
                            start_index >= len(offset_mapping)
                            or end_index >= len(offset_mapping)
                            or offset_mapping[start_index] is None
                            or len(offset_mapping[start_index]) < 2
                            or offset_mapping[end_index] is None
                            or len(offset_mapping[end_index]) < 2
                        ):
                            continue
                        # Don't consider answers with a length that is either < 0 or > max_answer_length.
                        if (
                            end_index < start_index
                            or end_index - start_index + 1 > max_answer_length
                        ):
                            continue
                        # Don't consider answer that don't have the maximum context available (if such information is
                        # provided).
                        if (
                            token_is_max_context is not None
                            and not token_is_max_context.get(str(start_index), False)
                        ):
                            continue

                        prelim_predictions.append(
                            {
                                "offsets": (
                                    offset_mapping[start_index][0],
                                    offset_mapping[end_index][1],
                                ),
                                "score": start_logits[start_index]
                                + end_logits[end_index],
                                "start_logit": start_logits[start_index],
                                "end_logit": end_logits[end_index],
                            }
                        )
            if version_2_with_negative and min_null_prediction is not None:
                # Add the minimum null prediction
                prelim_predictions.append(min_null_prediction)
                null_score = min_null_prediction["score"]

            # Only keep the best `n_best_size` predictions.
            predictions = sorted(
                prelim_predictions, key=lambda x: x["score"], reverse=True
            )[:n_best_size]

            # Add back the minimum null prediction if it was removed because of its low score.
            if (
                version_2_with_negative
                and min_null_prediction is not None
                and not any(p["offsets"] == (0, 0) for p in predictions)
            ):
                predictions.append(min_null_prediction)

            # Use the offsets to gather the answer text in the original context.
            context = example["context"]
            for pred in predictions:
                offsets = pred.pop("offsets")
                pred["text"] = context[offsets[0] : offsets[1]]

            # In the very rare edge case we have not a single non-null prediction, we create a fake prediction to avoid
            # failure.
            if len(predictions) == 0 or (
                len(predictions) == 1 and predictions[0]["text"] == ""
            ):
                predictions.insert(
                    0,
                    {
                        "text": "empty",
                        "start_logit": 0.0,
                        "end_logit": 0.0,
                        "score": 0.0,
                    },
                )

            # Compute the softmax of all scores
            scores = np.array([pred.pop("score") for pred in predictions])
            exp_scores = np.exp(scores - np.max(scores))
            probs = exp_scores / exp_scores.sum()

            # Include the probabilities in our predictions.
            for prob, pred in zip(probs, predictions):
                pred["probability"] = prob

            # Pick the best prediction. If the null answer is not possible, this is easy.
            if not version_2_with_negative:
                all_predictions[example["id"]] = predictions[0]["text"]
            else:
                # Otherwise we first need to find the best non-empty prediction.
                i = 0
                while predictions[i]["text"] == "":
                    i += 1
                best_non_null_pred = predictions[i]

                # Then we compare to the null prediction using the threshold.
                score_diff = (
                    null_score
                    - best_non_null_pred["start_logit"]
                    - best_non_null_pred["end_logit"]
                )
                scores_diff_json[example["id"]] = float(
                    score_diff
                )  # To be JSON-serializable.
                if score_diff > null_score_diff_threshold:
                    all_predictions[example["id"]] = ""
                else:
                    all_predictions[example["id"]] = best_non_null_pred["text"]

            # Make `predictions` JSON-serializable by casting np.float back to float.
            all_nbest_json[example["id"]] = [
                {
                    k: (
                        float(v)
                        if isinstance(v, (np.float16, np.float32, np.float64))
                        else v
                    )
                    for k, v in pred.items()
                }
                for pred in predictions
            ]

        # If we have an output_dir, let's save all those dicts.
        if output_dir is not None:
            if not os.path.isdir(output_dir):
                raise EnvironmentError(f"{output_dir} is not a directory.")

            prediction_file = os.path.join(
                output_dir,
                "predictions.json" if prefix is None else f"{prefix}_predictions.json",
            )
            nbest_file = os.path.join(
                output_dir,
                "nbest_predictions.json"
                if prefix is None
                else f"{prefix}_nbest_predictions.json",
            )
            if version_2_with_negative:
                null_odds_file = os.path.join(
                    output_dir,
                    "null_odds.json" if prefix is None else f"{prefix}_null_odds.json",
                )

            dprint(f"Saving predictions to {prediction_file}.")
            with open(prediction_file, "w") as writer:
                writer.write(json.dumps(all_predictions, indent=4) + "\n")
            dprint(f"Saving nbest_preds to {nbest_file}.")
            with open(nbest_file, "w") as writer:
                writer.write(json.dumps(all_nbest_json, indent=4) + "\n")
            if version_2_with_negative:
                dprint(f"Saving null_odds to {null_odds_file}.")
                with open(null_odds_file, "w") as writer:
                    writer.write(json.dumps(scores_diff_json, indent=4) + "\n")

        return all_predictions

    def post_processing_function(
        self,
        examples: Dataset,
        features: Dataset,
        predictions: list[np.ndarray],
        stage: str = "eval",
    ) -> dict[list[str, str]]:
        """Post-processing: we match the start logits and end logits to answers in
        the original context."""

        args = self.args
        predictions = self.postprocess_qa_predictions(
            examples=examples,
            features=features,
            predictions=predictions,
            version_2_with_negative=args.version_2_with_negative,
            n_best_size=args.n_best_size,
            max_answer_length=args.max_answer_length,
            null_score_diff_threshold=args.null_score_diff_threshold,
            output_dir=None,
            prefix=stage,
        )

        # Format the result to the format the metric expects.
        if args.version_2_with_negative:
            formatted_predictions = [
                {"id": k, "prediction_text": v, "no_answer_probability": 0.0}
                for k, v in predictions.items()
            ]
        else:
            formatted_predictions = [
                {"id": k, "prediction_text": v} for k, v in predictions.items()
            ]

        references = [
            {"id": ex["id"], "answers": ex[self.answer_column_name]} for ex in examples
        ]
        return EvalPrediction(predictions=formatted_predictions, label_ids=references)

    def create_and_fill_np_array(
        self,
        start_or_end_logits: list[np.ndarray],
        dataset: Dataset,
        max_len: int,
    ) -> np.ndarray:
        """
        Create and fill numpy array of size
        len_of_validation_data * max_length_of_output_tensor

        Args:
            start_or_end_logits(:obj:`tensor`):
                This is the output predictions of the model. We can only enter either
                start or end logits.
            eval_dataset: Evaluation dataset
            max_len(:obj:`int`):
                The maximum length of the output tensor. ( See the model.eval() part
                for more details )
        """

        step = 0
        # create a numpy array and fill it with -100.
        logits_concat = np.full((len(dataset), max_len), -100, dtype=np.float64)
        # Now since we have create an array now we will populate it with the outputs gathered using accelerator.gather_for_metrics
        for i, output_logit in enumerate(start_or_end_logits):  # populate columns
            # We have to fill it such that we have to take the whole tensor and replace it on the newly created array
            # And after every iteration we have to change the step

            batch_size = output_logit.shape[0]
            cols = output_logit.shape[1]

            if step + batch_size < len(dataset):
                logits_concat[step : step + batch_size, :cols] = output_logit
            else:
                logits_concat[step:, :cols] = output_logit[: len(dataset) - step]

            step += batch_size

        return logits_concat

    def run_warmup(self) -> None:
        """Run warmup cycle of compiled encoder model set for QuestionAnswering task."""

        dprint("Starting warm-up...")
        warmup_start_time = time.time()
        dataloader_for_compile = DataLoader(
            self.eval_dataset_for_model,
            shuffle=False,
            collate_fn=self.data_collator,
            batch_size=1,
        )
        first_batch = self.convert_batch_to_fms_style(
            next(iter(dataloader_for_compile))
        )
        self.model(**first_batch)
        dprint(f"Warmup completed in {time.time() - warmup_start_time:.1f} s\n---")

    def run_evaluation(self) -> None:
        """Run QuestionAnswering evaluation."""

        args = self.args
        eval_dataloader = self.eval_dataloader

        dprint(f"Running evaluation ({len(eval_dataloader)} samples)...")
        start_time = time.time()

        all_start_logits = []
        all_end_logits = []
        for step, batch in enumerate(eval_dataloader):
            with torch.no_grad():
                if args.verbose:
                    dprint(f"Step {step + 1} / {len(eval_dataloader)}")
                batch = self.convert_batch_to_fms_style(batch)
                batch = move_to_device(batch, args.device)
                start_logits, end_logits = self.model(**batch)
                all_start_logits.append(start_logits.to(torch.float16).cpu().numpy())
                all_end_logits.append(end_logits.to(torch.float16).cpu().numpy())
        eval_duration = time.time() - start_time
        dprint(
            f"Runtime: {eval_duration:.0f} s | "
            f"{eval_duration / len(eval_dataloader):.2f} s/batch | "
            f"{eval_duration / (len(eval_dataloader) * args.batch_size):.2f}"
            " s/sample "
            f"(tot = {len(eval_dataloader) * args.batch_size}, "
            f"bs = {args.batch_size})"
        )

        if rank == 0:
            # concatenate the numpy array
            max_len = max([x.shape[1] for x in all_start_logits])
            start_logits_concat = self.create_and_fill_np_array(
                all_start_logits,
                self.eval_dataset,
                max_len,
            )
            end_logits_concat = self.create_and_fill_np_array(
                all_end_logits,
                self.eval_dataset,
                max_len,
            )

            del all_start_logits
            del all_end_logits

            outputs_numpy = (start_logits_concat, end_logits_concat)
            prediction = self.post_processing_function(
                self.eval_examples,
                self.eval_dataset,
                outputs_numpy,
            )
            eval_metric = self.metric.compute(
                predictions=prediction.predictions,
                references=prediction.label_ids,
            )
            dprint(f"Evaluation metrics: {eval_metric}")


class EncoderMLMInfer:
    """Run MaskedLM task with encoder models."""

    def __init__(
        self,
        model: HFModelArchitecture,
        tokenizer: PreTrainedTokenizerBase,
        args: argparse.Namespace,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.args = args

    def process_eval_set(self) -> None:
        """Barebone function that sets up a single example prompt (for now)."""

        if not has_hf:
            raise ImportError(
                "MaskedLM Encoder requires transformers package but import "
                "was unsuccessful."
            )

        self.prompt = "the dog chased the cat while<mask> aggressively"

    def run_evaluation(self, warmup: bool = False) -> None:
        """Run evaluation cycle of compiled encoder model set for MaskedLM task.
        No output printout if warmup is True.
        """

        dprint(f"Starting evaluation ({warmup=})...")
        warmup_start_time = time.time()
        unmasker = pipeline(
            "fill-mask",
            model=self.model,
            tokenizer=self.tokenizer,
        )
        output = unmasker(self.prompt)
        dprint(f"Run completed in {time.time() - warmup_start_time:.1f} s\n---")
        if not warmup:
            dprint(f"{self.prompt}\nAnswers:")
            for ans in output:
                dprint(f"{ans['token_str']:10} | {ans['score']:6.4f}")


def run_encoder_eval_qa(
    model: nn.Module,  # FMS-style model
    tokenizer: PreTrainedTokenizerBase,
    args: argparse.Namespace,
) -> None:
    """Entry point to run QuestionAnswering Evaluation of encoder model.

    Processing based on pytorch example:
    https://github.com/huggingface/transformers/blob/main/examples/pytorch/...
    ...question-answering/run_qa_no_trainer.py
    """

    encoder_qa_infer = EncoderQAInfer(model, tokenizer, args)
    encoder_qa_infer.process_eval_set()
    if args.compile:
        encoder_qa_infer.run_warmup()
    encoder_qa_infer.run_evaluation()


def run_encoder_eval_mlm(
    model: HFModelArchitecture,  # model wrapped by to_hf_api
    tokenizer: PreTrainedTokenizerBase,
    args: argparse.Namespace,
) -> None:
    """Entry point to run evaluation of encoder models."""

    encoder_mlm_infer = EncoderMLMInfer(model, tokenizer, args)
    encoder_mlm_infer.process_eval_set()
    if args.compile:
        encoder_mlm_infer.run_evaluation(warmup=True)
    encoder_mlm_infer.run_evaluation()
