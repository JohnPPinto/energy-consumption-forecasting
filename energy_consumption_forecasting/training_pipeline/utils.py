from datetime import datetime
from typing import Literal, Optional

import wandb
from pydantic import validate_call

from energy_consumption_forecasting.utils import get_env_var

WANDB_API_KEY = get_env_var(key="WANDB_API_KEY")
WANDB_ENTITY = get_env_var(key="WANDB_ENTITY")
WANDB_PROJECT = get_env_var(key="WANDB_PROJECT")


@validate_call
def init_wandb_run(
    run_name: str,
    add_timestamp_to_run_name: bool = False,
    project_name: str = WANDB_PROJECT,
    entity: str = WANDB_ENTITY,
    group: Optional[str] = None,
    job_type: Optional[str] = None,
    resume: Optional[Literal["auto", "allow", "must", "never"]] = None,
    reinit: bool = False,
    run_id: Optional[str] = None,
) -> Optional[wandb.run]:
    """
    This function creates a wrapper to initiates the WandB experiment
    run with "wandb.init()" function for tracking the experiment.

    This can be used with a context manager like "with" block statement in python.

    Parameters
    ----------
    run_name: str
        A short name for the run, you can identify the run in the UI by this run name.

    add_timestamp_to_run_name: bool, default=False
        A timestamp is added at the end of the run name.

    project_name: str, default=WANDB_PROJECT
        The name of the project where newly created run will be send to, by default
        the project name takes the value from env variable with key "WANDB_PROJECT".

    entity: str, default=WANDB_ENTITY
        An entity is the username or a team name that is generated when an account
        is created in WandB, by default it takes the value from the env variable with
        key "WANDB_ENTITY".

    group: str or None, default=None
        A group name to group different runs into a specific group.

    job_type: str or None, default=None
        A job type to distinguish multiple different runs within the group.

    resume: Literal["auto", "allow", "must", "never"] or None, default=None
        Sets the resuming behavior for the run, only "auto", "allow", "must", "never"
        None are allowed as inputs.

    reinit: bool, default=False
        whether to allow multiple wandb.init() calls in the same process.

    run_id: str or None, default=None
        A unique id for this run , this can be used for resuming. It can not contains
        special characters like "/\#?%:".

    Returns
    -------
    wandb.run or None
        A wrapper around wandb.init() that returns the run object,
        can be used with the "with" block statement and None if
        the "wandb.finish" cleans up the run object.
    """

    if add_timestamp_to_run_name:
        run_name = f"{run_name}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"

    run = wandb.init(
        name=run_name,
        project=project_name,
        entity=entity,
        group=group,
        job_type=job_type,
        resume=resume,
        reinit=reinit,
        id=run_id,
    )

    return run
