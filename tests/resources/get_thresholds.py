import glob
import os
import numpy as np
import argparse
import os

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
    help="List of metrics separated by space. Eg.: diff_mean ce",
)
parser.add_argument(
    "--file_base",
    type=str,
    default="/tmp/aiu-fms-testing-utils/output",
    required=True,
    help="Path where the thresholds output from the generate_metrics.py script were stored.",
)
args = parser.parse_args()
models = [model.replace("/", "--") for model in args.models]
metrics = [metric for metric in args.metrics]
file_base = args.file_base

for model in models:
    for metric in metrics:
        path = os.path.join(file_base, f"{model}*{metric}*.csv")
        metric_files = glob.glob(path)

        metric_list = []
        for metric_file in metric_files:
            with open(metric_file, "r") as file:
                next(file)
                for line in file:
                    metric_list.append(float(line))
        print(f"found {len(metric_files)} metric files")
        print(model, metric, np.percentile(metric_list, 99.0))
