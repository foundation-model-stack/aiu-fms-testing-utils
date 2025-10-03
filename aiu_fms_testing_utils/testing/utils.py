from collections.abc import Iterable


def format_kwargs_to_string(**kwargs):
    """
    Turns kwargs into a str with variable names using `-`, variables separated by `_` and iterable separated by `,`
    """
    formatted_pairs = []
    for key, value in sorted(kwargs.items()):
        formatted_value = None
        if isinstance(value, str):
            formatted_value = value
        elif isinstance(value, Iterable):
            formatted_value = ",".join(map(str, value))
        elif value:
            formatted_value = str(value)
        # only append if formatted_value exists
        if formatted_value:
            # Keep previous convention of variable names with `-` instead of `_`
            formatted_pairs.append(
                f"{key.replace('_', '-')}-{formatted_value.replace('/', '--')}"
            )

    return "_".join(formatted_pairs)
