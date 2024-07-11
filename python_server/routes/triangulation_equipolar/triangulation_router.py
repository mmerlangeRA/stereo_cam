import json
import logging
import os
import sqlite3
from typing import Dict, List, Literal
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field,conlist
from python_server.utils.errors import INTERNAL_SERVER_ERROR_HTTPEXCEPTION, NOT_FOUND_HTTPEXCEPTION
from python_server.utils.tokens import verify_token
from python_server.components.triangulation_equipolar.main import AutoCalibrationRequest, TriangulationRequest, auto_calibrate_equipoloar, triangulate_equipolar_points
logger = logging.getLogger(__name__)

class TriangulationResponse(BaseModel):
    pointCam1: List[float]
    pointCam2: List[float]
    residuals: float

triangulation_router = APIRouter(prefix="/v1/triangulation")

@triangulation_router.post("/calibrate", tags=["equiolar triangulation"], response_model=conlist(item_type=float, min_length=6, max_length=6))
async def auto_calibrate_route(autoCalibrationRequest:AutoCalibrationRequest, request: Request) -> List[float]:
    try:
        calibration_params = auto_calibrate_equipoloar(autoCalibrationRequest)
        return calibration_params
    except FileNotFoundError as fnf_error:
        logger.error(f"File not found error: {fnf_error}")
        raise NOT_FOUND_HTTPEXCEPTION("File path not found")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise INTERNAL_SERVER_ERROR_HTTPEXCEPTION(f"An unexpected error occurred {e}")

@triangulation_router.post("/triangulate", tags=["equiolar triangulation"],response_model=TriangulationResponse)
async def triangulate_route(triangulationRequest:TriangulationRequest,request:Request)-> Dict[str, object]:
    try:
        point1, point2, residuals = triangulate_equipolar_points(triangulationRequest)
        return {
            "pointCam1": point1,
            "pointCam2": point2,
            "residuals": residuals
        }
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise INTERNAL_SERVER_ERROR_HTTPEXCEPTION(f"An unexpected error occurred {e}")


