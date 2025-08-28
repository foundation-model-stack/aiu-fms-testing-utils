import glob
import numpy as np
import argparse
import os
import re

import logging

import json

from aiu_fms_testing_utils.utils.metrics_utils import abs_diff_linalg_norm, list_mean

logger = logging.getLogger(__name__)
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s %(message)s")

parser = argparse.ArgumentParser(description="Script to get thresholds metrics")

parser.add_argument(
    "--models",
    type=str,
    default=[],
    nargs="+",
    required=True,
    help="List of models id separated by space. Eg.: ibm-granite/granite-20b-code-instruct-8k /tmp/models/granite-20b-code-cobol-v1",
)
parser.add_argument(
    "--metrics",
    type=str,
    default=[],
    nargs="+",
    required=True,
    help="List of metrics separated by space. Eg. for full model mode: diff_mean ce | Eg. for layers mode: abs_diff cos_sim_avg cos_sim_mean",
)
parser.add_argument(
    "--file_base",
    type=str,
    default="/tmp/aiu-fms-testing-utils/output",
    required=True,
    help="Path where the thresholds output from the generate_metrics.py script were stored.",
)
parser.add_argument(
    "--layer_io",
    action="store_true",
    help="Sets the metric generation mode to layers IO",
)
parser.add_argument(
    "--output_path",
    type=str,
    default="/tmp/aiu-fms-testing-utils/output",
    required=False,
    help="Path where the json thresholds output for the layers will be saved.",
)
args = parser.parse_args()
models = [model.replace("/", "--") for model in args.models]
metrics = [metric for metric in args.metrics]
file_base = args.file_base
layer_mode = args.layer_io if args.layer_io else False
generate_mode_pattern = r"\.(iter-)([0-9]+)"


def load_metric_file(file_path, layer_header, values):
    """
    Loads a metric file and returns its values as a list of floats.

    Args:
        file_path (str): The path to the metric file.
        layer_header (bool): Whether to skip the first three lines of the file. Default is False.
        values (list): Metrics values list

    Returns:
        list[float]: A list of metric values read from the file.
    """
    try:
        with open(file_path, "r") as file:
            if layer_header:
                for _ in range(3):
                    next(file)
            else:
                next(file)  # skip single header
            for line in file:
                values.append(float(line))
    except StopIteration:
        logger.info("Path empty or no more metric files found.")
        pass
    return values


for model in models:
    result_dict = {"model_id": model}
    for metric in metrics:
        metric_name = "_".join(metric.split("_")[:2]) if layer_mode else metric
        path = os.path.join(file_base, f"{model}*{metric_name}*.csv")
        metric_files = glob.glob(path)
        result_dict[metric] = {}
        if not layer_mode:
            metric_list = []
            for metric_file in metric_files:
                metric_list = load_metric_file(metric_file, layer_mode, metric_list)
            logger.info(f"found {len(metric_files)} metric files")
            metric_val = np.percentile(metric_list, 99.0)
            result_dict[metric] = metric_val
            logger.info(f"{model} {metric} = {metric_val}")
        else:
            layers = {}
            for metric_file in metric_files:
                layer_dict = {}
                metric_layer_list = []
                layer_name = metric_file.split("--")[-1].replace(
                    ".{}".format(metric_name), ""
                )
                layer_name = layer_name.replace(".csv", "")
                metric_layer_list = load_metric_file(
                    metric_file, layer_mode, metric_layer_list
                )
                if re.search(generate_mode_pattern, layer_name):
                    layer_name = re.sub(generate_mode_pattern, "", layer_name)
                    if layer_name not in layers.keys():
                        layers[layer_name] = metric_layer_list
                    else:
                        layers[layer_name].extend(metric_layer_list)
                    logger.debug(f"Output layer with generate mode {layer_name}")
                else:
                    layer_dict[layer_name] = metric_layer_list
                    logger.debug(f"Output layer {layer_name}")
                layers.update(layer_dict)
            logger.info(f"found {len(metric_files)} layers metric files")

            for key, layer in layers.items():
                layer = np.nan_to_num(layer, nan=0.0)
                if "abs_diff" in metric:
                    metric_val = abs_diff_linalg_norm(layer)
                    logger.info(f"Layer {key} abs_diff_linalg_norm = {metric_val}")
                    result_dict[metric][key] = metric_val
                elif "avg" in metric:
                    metric_avg = np.average(layer)
                    logger.info(f"Layer {key} {metric} = {metric_avg}")
                    result_dict[metric][key] = metric_avg
                elif "mean" in metric:
                    metric_mean = list_mean(layer)
                    logger.info(f"Layer {key} {metric} = {metric_mean}")
                    result_dict[metric][key] = metric_mean

    json_output_path = args.output_path if args.output_path else file_base
    f_result_path = os.path.join(json_output_path, f"{model}-thresholds.json")
    with open(f_result_path, "w") as fp:
        json.dump(result_dict, fp)
