import json
import logging
import os
import sqlite3
from typing import Dict, List, Literal
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field,conlist

from python_server.utils.errors import INTERNAL_SERVER_ERROR_HTTPEXCEPTION
from python_server.utils.tokens import verify_token
from python_server.components.triangulation.main import AutoCalibrationRequest, TriangulationRequest, auto_calibrate, triangulatePoints
logger = logging.getLogger(__name__)

class TriangulationResponse(BaseModel):
    pointCam1: List[float]
    pointCam2: List[float]
    residuals: float

triangulation_router = APIRouter(prefix="/v1/triangulation")

@triangulation_router.post("/calibrate", tags=["triangulation"], response_model=conlist(item_type=float, min_length=6, max_length=6))
async def auto_calibrate_route(autoCalibrationRequest:AutoCalibrationRequest, request: Request) -> List[float]:
    calibration_params = auto_calibrate(autoCalibrationRequest)
    return calibration_params

@triangulation_router.post("/triangulate", tags=["triangulation"],response_model=TriangulationResponse)
async def triangulate_route(triangulationRequest:TriangulationRequest,request:Request)-> Dict[str, object]:
     print("triangulate_route")
     point1, point2, residuals = triangulatePoints(triangulationRequest)
     print("triangulate_route done")
     print(point1, point2, residuals)
     return {
        "pointCam1": point1,
        "pointCam2": point2,
        "residuals": residuals
    }


