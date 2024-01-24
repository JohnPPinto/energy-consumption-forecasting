import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv


def get_env_var(env_path: Optional[str | Path] = None) -> dict:
    """
    Loads the environment variables and returns in dict.

    Parametrs
    ---------
    env_path: Optional[str | Path], default=None
        A path to the .env file.

    Returns
    -------
        A dict containing all the env variables.
    """
    if isinstance(env_path, str):
        env_path = Path(env_path)

    load_dotenv(dotenv_path=env_path, override=True)

    return dict(os.environ)
