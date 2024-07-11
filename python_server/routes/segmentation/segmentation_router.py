import json
import logging
import os
import sqlite3
from typing import Literal
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field
from python_server.utils.errors import INTERNAL_SERVER_ERROR_HTTPEXCEPTION, NOT_FOUND_HTTPEXCEPTION
from python_server.utils.tokens import verify_token
from python_server.components.pidnet_segementation.main import segment_image
from python_server.utils.types import Processed_file_response
logger = logging.getLogger(__name__)

class SegmentationRequest(BaseModel):
    image_name: str = Field(..., description="Name of the image to segment")


segmentation_router = APIRouter(prefix="/v1/segmentation")

@segmentation_router.post("/", tags=["segmentation"], response_model=Processed_file_response)
async def segmentate_image(segRequest: SegmentationRequest, request: Request) -> Processed_file_response:
    try:
        segmented_image_path = segment_image(segRequest.image_name)
        return Processed_file_response(public=segmented_image_path)
    except FileNotFoundError as fnf_error:
        logger.error(f"File not found error: {fnf_error}")
        raise NOT_FOUND_HTTPEXCEPTION("File path not found")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise INTERNAL_SERVER_ERROR_HTTPEXCEPTION(f"An unexpected error occurred {e}")
    


