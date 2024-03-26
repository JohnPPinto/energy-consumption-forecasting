import uvicorn

from backend.config import get_settings


def main() -> None:
    """
    A function to build the API application and connect with data.
    """

    uvicorn.run(
        "backend.application:get_app",
        workers=get_settings().WORKERS_COUNT,
        host=get_settings().HOST,
        port=get_settings().PORT,
        reload=get_settings().RELOAD,
        log_level=get_settings().LOG_LEVEL.value.lower(),
        factory=True,
    )


if __name__ == "__main__":
    main()
