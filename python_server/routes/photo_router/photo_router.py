import json
import logging
import os
import shutil
import sqlite3
from typing import List, Literal
from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile
from pathlib import Path
from pydantic import BaseModel, Field
from python_server.utils.errors import INTERNAL_SERVER_ERROR_HTTPEXCEPTION, NOT_FOUND_HTTPEXCEPTION
from python_server.utils.tokens import verify_token
from python_server.utils.path_helper import get_uploaded_photos_path, get_public_photo_path, get_static_path
from python_server.utils.types import Processed_file_response
logger = logging.getLogger(__name__)


photo_router = APIRouter(prefix="/v1/photo")

@photo_router.post("/", tags=["photos"],response_model=Processed_file_response)
async def upload_image(file: UploadFile = File(...))->Processed_file_response:
    try:
        file_location = Path(get_uploaded_photos_path(file.filename))
        with file_location.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        public_path = get_public_photo_path(file.filename)
        return Processed_file_response(public=public_path)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise INTERNAL_SERVER_ERROR_HTTPEXCEPTION(f"An unexpected error occurred {e}")

@photo_router.get("/files", tags=["photos"], response_model=List[str])
async def list_files() -> List[str]:
    try:
        static_folder = Path(get_static_path())        
        files = [f.name for f in static_folder.iterdir() if f.is_file()]
        return files
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise INTERNAL_SERVER_ERROR_HTTPEXCEPTION(f"An unexpected error occurred {e}")

@photo_router.delete("/files/{filename}", tags=["photos"])
async def delete_file(filename: str):
    try:
        file_path = Path(get_uploaded_photos_path(filename))
        if not file_path.exists() or not file_path.is_file():
            raise NOT_FOUND_HTTPEXCEPTION("File not found")

        file_path.unlink()
        return {"detail": f"File '{filename}' deleted successfully"}
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise INTERNAL_SERVER_ERROR_HTTPEXCEPTION(f"An unexpected error occurred {e}")