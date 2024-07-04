"""FastAPI app creation, logger configuration and main API routes."""
from python_server.di import global_injector
from python_server.launcher import create_app


app = create_app(global_injector)
