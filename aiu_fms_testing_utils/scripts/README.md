# Scripts for using Foundation Model Stack (FMS) on AIU hardware

The scripts provided here allow you to run FMS on AIU device for a variety of models.

Let's look at some of the example usage below.

## How to run an encoder model
The script [run_encoders.py](https://github.com/foundation-model-stack/aiu-fms-testing-utils/blob/main/scripts/run_encoders.py) allows you to run an encoder model. Its usage is demonstrated below.

```bash
# run RoBERTa on AIU
python3 run_encoders.py --architecture=hf_pretrained --model_path=/home/senuser/roberta --tokenizer=/home/senuser/roberta --unfuse_weights --device_type=aiu --compile --compile_dynamic

# run RoBERTa on CPU
python3 run_encoders.py --architecture=hf_pretrained --model_path=/home/senuser/roberta --tokenizer=/home/senuser/roberta --unfuse_weights --device_type=cpu
```

## How to run a decoder model
The script [inference.py](https://github.com/foundation-model-stack/aiu-fms-testing-utils/blob/main/scripts/inference.py) allows you to run a decoder model. Its usage is demonstrated below for various Llama and Granite models.

```bash
# run 194m on AIU
python3 inference.py --architecture=hf_pretrained --model_path=/home/senuser/llama3.194m --tokenizer=/home/senuser/llama3.194m --unfuse_weights --min_pad_length 64 --device_type=aiu --max_new_tokens=5 --compile --default_dtype=fp16 --compile_dynamic

# run 194m on CPU
python3 inference.py --architecture=hf_pretrained --model_path=/home/senuser/llama3.194m --tokenizer=/home/senuser/llama3.194m --unfuse_weights --min_pad_length 64 --device_type=cpu --max_new_tokens=5 --default_dtype=fp32

# run 7b on AIU
python3 inference.py --architecture=hf_pretrained --model_path=/home/senuser/llama2.7b --tokenizer=/home/senuser/llama2.7b --unfuse_weights --min_pad_length 64 --device_type=aiu --max_new_tokens=5 --compile --default_dtype=fp16 --compile_dynamic

# run 7b on CPU
python3 inference.py --architecture=hf_pretrained --model_path=/home/senuser/llama2.7b--tokenizer=/home/senuser/llama2.7b --unfuse_weights --min_pad_length 64 --device_type=cpu --max_new_tokens=5 --default_dtype=fp32

# run gpt_bigcode (granite) 3b on AIU
python3 inference.py --architecture=gpt_bigcode --variant=ibm.3b --model_path=/home/senuser/gpt_bigcode.granite.3b/*00002.bin --model_source=hf --tokenizer=/home/senuser/gpt_bigcode.granite.3b --unfuse_weights --min_pad_length 64 --device_type=aiu --max_new_tokens=5 --prompt_type=code --compile --default_dtype=fp16 --compile_dynamic

# run gpt_bigcode (granite) 3b on CPU
python3 inference.py --architecture=gpt_bigcode --variant=ibm.3b --model_path=/home/senuser/gpt_bigcode.granite.3b/*00002.bin --model_source=hf --tokenizer=/home/senuser/gpt_bigcode.granite.3b --unfuse_weights --min_pad_length 64 --device_type=cpu --max_new_tokens=5 --prompt_type=code --default_dtype=fp32
```

## How to run tensor parallel
The [small-toy.py](https://github.com/foundation-model-stack/aiu-fms-testing-utils/blob/main/scripts/small-toy.py) is a slimmed down version of the Big Toy model. The purpose of this script is to demonstrate how to run a tensor parallel model with the FMS on AIU hardware.

The `--nproc-per-node` command line option controls the number of AIUs to use (number of parallel processes).

```bash
# 1 AIU (sequential)
# Inductor (CPU) backend (default)
torchrun --nproc-per-node 1 ./small-toy.py
# AIU backend
torchrun --nproc-per-node 1 ./small-toy.py --backend aiu

# 2 AIUs (tensor parallel)
# Inductor (CPU) backend (default)
torchrun --nproc-per-node 2 ./small-toy.py
# AIU backend
torchrun --nproc-per-node 2 ./small-toy.py --backend aiu
```

## How to validate models
The [validation.py](https://github.com/foundation-model-stack/aiu-fms-testing-utils/blob/main/scripts/validation.py) provides an example of validating models on AIU through comparisons to other devices. Its usage is demonstrated below for various cases.

```bash
# Run a llama 194m model, grab the example inputs in the script, generate validation tokens on cpu, validate token equivalency: 
python3 scripts/validation.py --architecture=hf_pretrained --model_path=/home/devel/models/llama-194m --tokenizer=/home/devel/models/llama-194m --unfuse_weights --batch_size=1 --min_pad_length=64 --max_new_tokens=10 --compile_dynamic

# Run a llama 194m model, grab the example inputs in a folder, generate validation tokens on cpu, validate token equivalency:
python3 scripts/validation.py --architecture=hf_pretrained --model_path=/home/devel/models/llama-194m --tokenizer=/home/devel/models/llama-194m --unfuse_weights --batch_size=1 --min_pad_length=64 --max_new_tokens=10 --prompt_path=/home/devel/aiu-fms-testing-utils/prompts/test/*.txt --compile_dynamic

# Run a llama 194m model, grab the example inputs in a folder, grab validation text from a folder, validate token equivalency (will only validate up to max(max_new_tokens, tokens_in_validation_file)):
python3 scripts/validation.py --architecture=hf_pretrained --model_path=/home/devel/models/llama-194m --tokenizer=/home/devel/models/llama-194m --unfuse_weights --batch_size=1 --min_pad_length=64 --max_new_tokens=10 --prompt_path=/home/devel/aiu-fms-testing-utils/prompts/test/*.txt --validation_files_path=/home/devel/aiu-fms-testing-utils/prompts/validation/*.txt --compile_dynamic

# Validate a reduced size version of llama 8b
python3 scripts/validation.py --architecture=hf_configured --model_path=/home/devel/models/llama-8b --tokenizer=/home/devel/models/llama-8b --unfuse_weights --batch_size=1 --min_pad_length=64 --max_new_tokens=10 --extra_get_model_kwargs nlayers=3 --compile_dynamic
```

To run a logits-based validation, pass `--validation_level=1` to the validation script. This will check for the logits output to match at every step of the model through cross-entropy loss. You can control the acceptable threshold with `--logits_loss_threshold`.

## Setup the environment for reporting resource usage

When running `drive_paged_programs.py` you may want to see how much CPU and memory usage is
happening. This is done using Prometheus, thus if you are running in a container environment (non-OpenShift), you want to set up a simple Prometheus server to start collecting these metrics. To do this, do the following:

1. Run `podman network create promnet`
2. Run `podman run -d --name node-exporter --network promnet quay.io/prometheus/node-exporter:latest`
3. Create a file called `prometheus.yml` that has the following contents:

```yaml
global:
  scrape_interval: 5s

scrape_configs:
  - job_name: "node"
    static_configs:
      - targets: ["node-exporter:9100"]
```

4. Run `podman run -d --name prometheus --network promnet -p 9091:9090   -v "$PWD/prometheus.yml:/etc/prometheus/prometheus.yml:Z"   quay.io/prometheus/prometheus:latest   --config.file=/etc/prometheus/prometheus.yml`
5. Check the status of the server by running `curl -s "http://localhost:9091/api/v1/targets" | python3 -m json.tool | grep health` and ensuring that "health" says "up".
6. When you are about to run DPP, run `export PROMETHEUS_URL="http://localhost:9091"`

If you are running in OpenShift, the aformentioned instructions are not neccessary and instead, you are going to want to set `PROMETHEUS_URL` to an OpenShift route that already has Prometheus set up. Additionally, you are going to want to set `PROMETHEUS_API_KEY` to your OpenShift OAuth token if the Prometheus instance on the cluster is protected. You can get this token by running `oc whoami -t`.

When actually running a DPP test, you are going to want to set the `--report_resource_utilization` flag to see outputs. Regardless if you have this flag set or if you do not have Prometheus installed or any of the environment variables set, DPP should always run. These instructions are simply just to see resource utilization outputs.

Sample test to run with resource utilization outputs:

```bash
torchrun --nproc-per-node=4 aiu-fms-testing-utils/scripts/drive_paged_programs.py --model_variant=/ibm-granite/granite-3.3-8b-instruct --program_criteria_json_path=path/to/program_criterion.json --dataset_type=sharegpt --skip_validation --programs "*:0,<8192" --prioritize_large_batch_sizes --enforce_homogeneous_prompt_programs --prefill_chunk_size=1024 --dataset_path=ShareGPT_V3_unfiltered_cleaned_split.json --report_resource_utilization
```
