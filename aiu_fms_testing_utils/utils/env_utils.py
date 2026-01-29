import os
from contextlib import contextmanager
from typing import Optional


@contextmanager
def scoped_environ(updates: dict[str, Optional[str]]):
    """
    Temporarily set environment variables.
    Restores original values on exit.

    updates:
      key -> value
      value=None means unset the variable
    """
    old_env = {}

    try:
        # Save old values and apply updates
        for key, value in updates.items():
            old_env[key] = os.environ.get(key)
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = str(value)
        yield
    finally:
        # Restore original environment
        for key, old_value in old_env.items():
            if old_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old_value
