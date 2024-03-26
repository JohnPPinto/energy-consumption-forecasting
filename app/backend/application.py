from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.config import get_settings
from backend.endpoint import api_router


def get_app() -> FastAPI:
    """
    This function creates FastAPI application and can be used to connect
    with the defined endpoints.
    """

    app = FastAPI(
        title=get_settings().PROJECT_NAME,
        docs_url=f"/api/{get_settings().API_VERSION}/docs",
        redoc_url=f"/api/{get_settings().API_VERSION}/redoc",
        openapi_url=f"/api/{get_settings().API_VERSION}/openapi.json",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(router=api_router, prefix=f"/api/{get_settings().API_VERSION}")

    return app
