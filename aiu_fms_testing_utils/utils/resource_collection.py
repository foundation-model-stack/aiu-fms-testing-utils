# Imports
import os
from datetime import datetime, timezone

from aiu_fms_testing_utils.utils.aiu_setup import dprint
try:
    from prometheus_api_client import PrometheusConnect
except Exception:
    print("WARNING: Cannot import `prometheus_api_client`. Make sure the package is installed if you are trying to report resource utilization.")


def instantiate_prometheus():
    """
    Top-level method that will instantiate the Prometheus Client to collect
    resource usage metrics.

    Returns:
    - client: the instantiated Prometheus client.
    """

    client = None
    try:
        # Get required env variables
        connection_url = os.environ["PROMETHEUS_URL"]
        api_token = os.environ.get("PROMETHEUS_API_KEY")

        # Define necessary headers
        request_headers = {"Authorization": f"Bearer {api_token}"} if api_token else None

        client = PrometheusConnect(url=connection_url, headers=request_headers, disable_ssl=True)

    except Exception as e:
        print(f"WARNING: Cannot instantiate Prometheus. Make sure PROMETHEUS_URL and PROMETHEUS_API_KEY are set in your environment if you are trying to collect resource metrics. Error: {e}")

    return client


def get_value(given_res, query_type="static"):
    """
    Helper method to get the given value from a Prometheus response

    Args:
    - given_res: The response object obtained from the Prometheus client that has our value.
    - query_type: The type of query we are processing, "static" or "range"

    Returns:
    - value: the value for the given resource metric we want to report that was obtained from
    the response, represented as a float if present, otherwise None.
    """

    # Iterate through to save our output to a list
    values = []
    value = None
    if query_type == "static":  ## For start/end reads
        for series in given_res or []:
            try:
                values.append(float(series["value"][1]))
            except Exception:
                pass
        value = values[0] if values else None
    
    else:  ## For peak reads
        for series in given_res or []:
            for timestamp, val in series.get("values", []):
                try:
                    values.append(float(val))
                except Exception:
                    pass
        value = max(values) if values else None

    return value


def get_static_read(client, recorded_time):
    """
    Top-level method that will get a read on CPU and Memory usage give a single
    moment in time.

    Args:
    - client: the Prometheus client to use to get our metrics.
    - recorded_time: the time that we want to get the metric read at. 

    Returns:
    - cpu_value: this is the reported value for percentage of CPU usage at the given
    recorded time.
    - mem_value: this is the reported value for memory usage at the given
    recorded time in gigabytes.
    """

    cpu_value = None
    mem_value = None
    if client is not None:

        # Make the request for CPU and Mem
        cpu_query = '100 * (1 - avg(rate(node_cpu_seconds_total{mode="idle"}[2m])))'
        mem_query = '(node_memory_MemTotal_bytes - node_memory_MemAvailable_bytes) / 1024 / 1024 / 1024'
        cpu_response = client.custom_query(query=cpu_query, params={"time": recorded_time.timestamp()})
        mem_response = client.custom_query(query=mem_query, params={"time": recorded_time.timestamp()})

        ## Get the CPU & Mem metrics out of the response
        cpu_value = get_value(cpu_response)
        mem_value = get_value(mem_response)

    return cpu_value, mem_value


def get_peak_read(client, start, end):
    """
    Top-level method that will get the peak resource usage during a given interval.

    Args:
    - client: the Prometheus client to use to get our metrics.
    - start: the recorded start time for the interval.
    - end: the recorded end time for the interval.

    Returns:
    - peak_cpu_value: this is the peak reported value for percentage of CPU usage over the
    given interval.
    - peak_mem_value: this is the peak reported value for memory usage over the given interval
    in gigabytes.

    """

    peak_cpu_value = None
    peak_mem_value = None
    if client is not None:

        # Make the request for CPU and Mem
        cpu_query = '100 * (1 - avg(rate(node_cpu_seconds_total{mode="idle"}[2m])))'
        mem_query = '(node_memory_MemTotal_bytes - node_memory_MemAvailable_bytes) / 1024 / 1024 / 1024'
        cpu_response = client.custom_query_range(
            query=cpu_query, start_time=start, end_time=end, step="3s"
        )
        mem_response = client.custom_query_range(
            query=mem_query, start_time=start, end_time=end, step="3s"
        )

        ## Get the CPU & Mem metrics out of the response
        peak_cpu_value = get_value(cpu_response, "range")
        peak_mem_value = get_value(mem_response, "range")

    return peak_cpu_value, peak_mem_value


def timestamp_print(given_string):
    """
    Helper method that will add a timestamp before the given string that needs to be
    printed.

    Args:
    - given_string: the string that is to be printed with the timestamp.
    """

    timestamp = datetime.now().strftime("%Y-%m-%d:%H:%M:%S")
    print(f"[{timestamp}] {given_string}")


def print_comp_resource_metrics(cpu_val, mem_val, stage, step):
    """
    Helper method that will do a timestamp print for a specific step to report resource
    usage.

    Args:
    - cpu_val: the value for CPU usage as a percentage that we want to print.
    - mem_val: the value for memory usage in gigabytes we want to print.
    - stage: The stage of the step we are in, either "peak" or "started".
    - step: The step that we performing in the script, either "compilation" or "inference".
    """

    if stage != "peak":
        if cpu_val is None or mem_val is None:
            timestamp_print(f"{step} {stage}")
        else:
            timestamp_print(f"{step} {stage} - CPU: {cpu_val:.2f}%, Memory: {mem_val:.2f} GB")

    elif cpu_val is not None and mem_val is not None:
        dprint(f"Peak Resource Utilization - CPU: {cpu_val:.2f}%, Memory: {mem_val:.2f} GB")


def print_step(p, step, stage, start_time=None):
    """
    Print function to print out when a specific stage starts and ends,
    as well as reporting resource usage if enabled.

    Args:
    - p: the Prometheus profile client to resource utilization collection.
    - step: string denoting what step we are at ("inference" or "compilation").
    - stage: string denoting what stage of the step we are at ("started" or "completed").
    - start_time: datetime object that denotes when the step started (optional).

    Returns:
    - recorded_time: the time that was recorded when getting a metric read. Returned for
    scenarios where we need to use the recorded time in a later step (i.e completed stages).
    """

    ## Get metric read
    recorded_time = datetime.now(timezone.utc)
    cpu_usage, mem_usage = get_static_read(p, recorded_time)
    print_comp_resource_metrics(cpu_usage, mem_usage, step, stage)

    ## Get and print the peak usage
    if start_time is not None:
        peak_cpu_inference_cpu, peak_mem_inference_cpu = get_peak_read(p, start_time, recorded_time)
        print_comp_resource_metrics(peak_cpu_inference_cpu, peak_mem_inference_cpu, "peak", stage)

    return recorded_time
