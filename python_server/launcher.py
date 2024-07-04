"""FastAPI app creation, logger configuration and main API routes."""
import logging
import os

from fastapi import Depends, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from injector import Injector
from python_server.components.triangulation.database.main import init_db
from python_server.routes.segmentation.segmentation_router import segmentation_router 
from python_server.routes.triangulation.triangulation_router import triangulation_router 

from python_server.settings.settings import Settings
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager

from src.utils.path_utils import create_static_folder

logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield

def create_app(root_injector: Injector) -> FastAPI:
    
    # Start the API
    async def bind_injector_to_request(request: Request) -> None:
        request.state.injector = root_injector

    app = FastAPI(lifespan=lifespan, dependencies=[Depends(bind_injector_to_request)])

    create_static_folder()

    app.mount("/static", StaticFiles(directory="static"), name="static")
    app.include_router(triangulation_router)
    app.include_router(segmentation_router)
    settings = root_injector.get(Settings)

    if settings.server.cors.enabled:
        logger.debug("Setting up CORS middleware")
        app.add_middleware(
            CORSMiddleware,
            allow_credentials=settings.server.cors.allow_credentials,
            allow_origins=settings.server.cors.allow_origins,
            allow_origin_regex=settings.server.cors.allow_origin_regex,
            allow_methods=settings.server.cors.allow_methods,
            allow_headers=settings.server.cors.allow_headers,
        )

    return app
