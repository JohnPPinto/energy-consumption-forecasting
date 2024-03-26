import enum
from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class LogLevel(str, enum.Enum):
    """
    Levels present in the logging package
    """

    NOTSET = "NOTSET"
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    FATAL = "FATAL"


class Settings(BaseSettings):
    """
    This class gets or sets configuration for the application project.
    Configuration values can be modified using the environment variable file.
    The parameters in the "app.env" file needs to hae a prefix of "APP_API_".
    """

    model_config = SettingsConfigDict(
        env_file="./secrets/app/app.env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # API project configuration
    PROJECT_NAME: str = "Denmark Energy Consumption Forecasting API"
    API_VERSION: str = "v1"
    HOST: str = "0.0.0.0"
    PORT: int = 8001
    LOG_LEVEL: LogLevel = LogLevel.INFO

    # Uvicorn configuration
    WORKERS_COUNT: int = 1
    RELOAD: bool = False

    # Google cloud configuration
    GCP_PROJECT: str = Field(alias="GOOGLE_CLOUD_PROJECT")
    GCP_BUCKET_NAME: str = Field(alias="GOOGLE_CLOUD_BUCKET_NAME")
    GCP_SERVICE_ACCOUNT_FILE: str = Field(
        alias="GOOGLE_CLOUD_SERVICE_ACCOUNT_JSON_PATH"
    )


@lru_cache()
def get_settings():
    return Settings()


if __name__ == "__main__":

    print(Settings().model_dump())
