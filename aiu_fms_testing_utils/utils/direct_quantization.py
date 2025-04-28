# Standard
from pathlib import Path
from tqdm import tqdm
import argparse
import os
import time

# Third Party
from torch.utils.data import DataLoader  # [R]
from transformers import (  # [R]
    default_data_collator,
    DataCollatorWithPadding,
    EvalPrediction,
    RobertaForQuestionAnswering,
    RobertaForMaskedLM,
    RobertaTokenizerFast,
    pipeline,
)
import torch

# Local Packages
from fms_mo import qconfig_init, qmodel_prep  # [R]
from fms_mo.quant.ptq import dq_llm, get_act_scales  # [R]
from fms_mo.utils.utils import prepare_input  # [R]
from utils.roberta_int8_utils import (  # [R] change this
    validate_arguments,
    get_wikitext2,
    use_default_qcfg,
    process_state_dict,
    mask_examples,
    dequantize_int8_weights,
)


QUANTIZED_LAYERS_ROBERTA = [
    "attention.self.query",
    "attention.self.key",
    "attention.self.value",
    "attention.output.dense",
    "intermediate.dense",
    "output.dense",
]

# TODO: change print to dprint
# TODO: add LLM DQ
# TODO: load wikitext using FMS-MO instead of custom function

def run_dq_roberta(args: argparse.Namespace):
    """Run INT8 Direct Quantization for RoBERTa.
    """

    #-------------
    # Instantiate HF RoBERTa FP16
    #-------------
    print("* Begin Direct Quantization (DQ) process.")
    torch.set_default_dtype(torch.float16)
    tokenizer = RobertaTokenizerFast.from_pretrained(args.tokenizer)
    fp16_model_path = args.fp16_ckpt_path if args.fp16_ckpt_path else 'roberta-base'
    if args.architecture == "roberta":
        model = RobertaForMaskedLM.from_pretrained(
            fp16_model_path,
            torch_dtype=torch.float16,
        )
    elif args.architecture == "roberta_question_answering":
        model = RobertaForQuestionAnswering.from_pretrained(
            fp16_model_path,
            torch_dtype=torch.float16,
        )
    else:
        raise NotImplementedError(
            f"Variant {args.architecture} is not supported for Direct Quantization"
        )
    model.to("cpu")
    print("* FP16 model loaded to CPU.")

    train_dataset, test_dataset = get_wikitext2(tokenizer)
    dq_dataloader = DataLoader(
        train_dataset[:args.num_dq_samples],
        shuffle=True,
        collate_fn=default_data_collator,
        batch_size=1,
    )
    print(f"* Dataset for DQ loaded (samples = {len(dq_dataloader.dataset)}).")

    #-------------
    # Set fms_mo configuration
    #-------------
    qcfg = qconfig_init(recipe=args.int8_qcfg_path, args=args)

    # preferred method is to update qconfig from recipe, providing --int8_qcfg_path
    # but the following will set some defaults if config json is not passed
    if not args.int8_qcfg_path:
        print("* Using a default quantization configuration for missing parameters.")
        qcfg = use_default_qcfg(qcfg)
        qcfg["logger"] = print
        qcfg["qw_mode"] = "maxperCh" if args.weight_per_channel else "max"
        if args.activ_quant_type == "per_token":
            qcfg["qa_mode"] = "pertokenmax"
        elif args.activ_quant_type == "per_tensor_symm":
            qcfg["qa_mode"] = "maxsym"
        else:
            qcfg["qa_mode"] = "max"
        qcfg["a_init_method"] = "max"
        qcfg["qw_mode_calib"] = "max"
        qcfg["qa_mode_calib"] = "max"

    if args.verbose:
        print("=" * 60)
        print("QUANTIZATION CONFIGURATION")
        print("\n".join(f"{k:60} {v}" for k,v in qcfg.items() if not isinstance(v, dict)))

    #-------------
    # Prepare inputs as list and generate quantized model with fms_mo
    # This is not an FMS model. fms_mo model can run Direct Quantization
    #-------------
    examples = None
    examples_for_prep = None
    if qcfg["qmodel_calibration"]:
        if args.activ_quant_type == "per_tensor_asymm":
            print("=" * 60)
            print(f"qmodel_calibration = {qcfg['qmodel_calibration']}")
            print(f"qmodel_calibration_new = {qcfg['qmodel_calibration_new']}")
            raise NotImplementedError(
                "Direct Quantization (DQ) using `qmodel_calibration` is not compatible "
                "with INT8 asymmetric quantization of activations in fms-mo. "
                "Please pass `qmodel_calibration_new` argument instead."
            )
        examples = qcfg["qmodel_calibration"]
    elif qcfg["qmodel_calibration_new"]:
        examples = qcfg["qmodel_calibration_new"]
    if examples:
        examples_for_prep = [next(iter(dq_dataloader)) for _ in range(examples)]

    #-------------
    # Prepare quantized model using fms_mo
    #-------------
    print("=" * 60)
    print(f"* Begin preparation of quantized model.")
    if qcfg["qmodel_calibration"]:
        print("* Calibration to be applied during this preparation step.")
    prep_time_start = time.time()
    qmodel_prep(
        model,
        examples_for_prep,
        qcfg,
        dev="cpu",  # always run Direct Quantization on CPU, not AIU
        use_layer_name_pattern_matching=False,
        save_fname='roberta-base-w8a8',
    )
    if qcfg["qmodel_calibration"]:
        print(
            "* Quantized model has been instantiated and pre-calibrated "
            f"(took {time.time() - prep_time_start:.1f} s)."
        )
    else:
        print(
            "* Quantized model has been instantiated and needs calibration "
            f"(took {time.time() - prep_time_start:.1f} s)."
        )

    #-------------
    # Apply smoothquant
    #-------------
    if qcfg['smoothq']:
        sq_time_start = time.time()
        print("* Being applying SmoothQuant scales.")
        assert qcfg['smoothq'] == True, "doing smoothq"
        if not os.path.exists(qcfg['act_scale_path']):
            print(
                "generate new smoothq activation scales "
                f"at {qcfg['act_scale_path']}"
            )
            smoothq_alpha_requested = None
            if qcfg["smoothq_alpha"] != 0:
                smoothq_alpha_requested = qcfg["smoothq_alpha"]
                qcfg["smoothq_alpha"] = 0
                print("[WARNNG] using smoothq_alpha = 0 for scale generation")
            act_scales = get_act_scales(model, dq_dataloader, qcfg, device="cpu")
            torch.save(act_scales, qcfg['act_scale_path'])
            if smoothq_alpha_requested:
                qcfg["smoothq_alpha"] = smoothq_alpha_requested
                print(f"smoothq_alpha set back to {qcfg['smoothq_alpha']}")
        else:
            print(
                f"using smoothq activation scales from {qcfg['act_scale_path']}"
            )
            act_scales = torch.load(qcfg['act_scale_path'], map_location='cpu')

        dq_llm(model, act_scales, qcfg)
        print(f"* SmoothQuant scales applied (took = {time.time() - sq_time_start:.1f} s).")
        print("=="*20)
    else:
        print("* SmoothQuant is DISABLED.")

    #-------------
    # Run calibration = Direct Quantization DQ
    #-------------
    if qcfg['qmodel_calibration_new'] > 0:
        calib_time_start = time.time()
        print("* Begin calibration of activation quantized parameters.")
        pbar = tqdm(
            dq_dataloader,
            desc="* Calibration progress",
            total = qcfg['qmodel_calibration_new']
            )
        for data_mb, _ in zip(pbar, range(qcfg['qmodel_calibration_new'])):
            data_mb = prepare_input(
                device=model.device,
                data=data_mb,
            )
            with torch.no_grad():
                model(**data_mb)
        print(f"* Calibration completed (took = {time.time() - calib_time_start:.1f} s).")

    if args.verbose:
        print("=" * 60)
        print("* PARAMETERS")
        print("\n".join(
            f"{k:80} {str(list(v.size())):15} {v.dtype}"
            for k,v in model.named_parameters()
        ))
        print("* BUFFERS")
        print("\n".join(
            f"{k:80} {str(list(v.size())):15} {v.dtype}"
            for k,v in model.named_buffers()
        ))

    #-------------
    # Save checkpoint with integer weights (AIU requirement)
    #-------------
    keys_to_ignore = [
        "num_module_called",
        "smoothq_act_scale",
        "smoothq_alpha",
        "calib_counter",
        "obsrv_clipval",
        "obsrv_clipvaln",
        "obsrv_w_clipval",
    ]

    print(f"Begin processing model state dictionary for saving.")
    new_sd = process_state_dict(
        model=model,
        quantized_layers=QUANTIZED_LAYERS_ROBERTA,
        keys_to_ignore=keys_to_ignore,
        verbose=args.verbose,
    )

    task = "mlm" if args.architecture == "roberta" else "qa"
    smoothq_str = qcfg['smoothq_alpha'] if qcfg['smoothq'] else "no"
    save_path = str(
        Path(args.output_path) /
        f"roberta-base_{task}_w8-{qcfg['qw_mode']}_a8-{qcfg['qa_mode']}"
        f"_bmm32_smoothq-{smoothq_str}_dq.pt"
    )
    torch.save(new_sd, save_path)
    print(f"Model saved to {save_path}")

    tokenizer.save_pretrained(args.output_path)
    print(f"Tokenizer saved to {args.output_path}")