import json
import logging
import os
import sqlite3
from typing import Literal
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field
from python_server.utils.errors import INTERNAL_SERVER_ERROR_HTTPEXCEPTION
from python_server.utils.tokens import verify_token
from python_server.components.pidnet_segementation.main import segment_image
logger = logging.getLogger(__name__)

class SegmentationRequest(BaseModel):
    image_name: str = Field(..., description="Name of the image to segment")

class SegmentationResponse(BaseModel):
    segmentedImage: str

segmentation_router = APIRouter(prefix="/v1/segmentation")

@segmentation_router.post("/", tags=["segmentation"], response_model=SegmentationResponse)
async def router_autocalibrate_EAC_path(segRequest: SegmentationRequest, request: Request) -> SegmentationResponse:
    segmented_image_path = segment_image(segRequest.image_name)
    return SegmentationResponse(segmentedImage=segmented_image_path)


