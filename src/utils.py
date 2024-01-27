import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv


def get_env_var(
    key: str,
    default_value: str = None,
    env_path: Optional[str | Path] = None,
) -> Optional[str]:
    """
    Loads the environment variables and returns the value for the requested key.

    Parameters
    ----------
    key: str
        The key of the requested environment variable.

    default_value: str, default=None
        The value to be returned if environment variable is not found.

    env_path: str or Path or None, default=None
        A path to the .env file.

    Returns
    -------
    value: str or None
        The value of the environment variable that is requested.
    """
    if isinstance(env_path, str):
        env_path = Path(env_path)

    load_dotenv(
        dotenv_path=env_path,
        override=True,
    )

    if default_value is None:
        value = os.getenv(key=key)
    else:
        value = os.getenv(key=key, default=default_value)

    return value
