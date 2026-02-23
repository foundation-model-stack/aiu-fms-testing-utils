import os

from prometheus_api_client import PrometheusConnect


def instantiate_prometheus():
    """
    Top-level method that will instantiate the Prometheus Client to collect
    resource usage metrics.

    Returns:
    - PrometheusConnect(url=connection_url, headers=request_headers): the instantiated
    Prometheus client.
    """

    # Get required env variables
    connection_url = os.environ["PROMETHEUS_URL"]
    api_token = os.environ.get("PROMETHEUS_API_KEY")

    # Define necessary headers
    request_headers = {"Authorization": f"Bearer {api_token}"} if api_token else None

    return PrometheusConnect(url=connection_url, headers=request_headers)


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

    # Make the request for CPU and Mem
    cpu_query = '100 * (1 - avg(rate(node_cpu_seconds_total{mode="idle"}[2m])))'
    mem_query = '100 * (1 - (node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes))'
    cpu_response = client.custom_query(query=cpu_query, params={"time", recorded_time.timestamp()})
    mem_response = client.custom_query(query=mem_query, params={"time", recorded_time.timestamp()})

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

    # Make the request for CPU and Mem
    cpu_query = '100 * (1 - avg(rate(node_cpu_seconds_total{mode="idle"}[2m])))'
    mem_query = '100 * (1 - (node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes))'
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
