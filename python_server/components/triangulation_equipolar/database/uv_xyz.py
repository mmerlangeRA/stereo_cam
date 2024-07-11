import json
from typing import List
from sqlalchemy import Column, Integer, Float, String, and_
from sqlalchemy.ext.declarative import declarative_base
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from sqlalchemy.types import TypeDecorator, TEXT
from sqlalchemy import func
from python_server.components.triangulation_equipolar.database.base import Base

class JsonType(TypeDecorator):
    impl = TEXT

    def process_bind_param(self, value, dialect):
        if value is None:
            return None
        return json.dumps(value)

    def process_result_value(self, value, dialect):
        if value is None:
            return None
        return json.loads(value)

class UV_XYZ_Schema(BaseModel):
    image_name: str
    values:List[float]

    class Config:
        from_attributes = True

class UV_XYZ(Base):
    __tablename__ = 'uv_xyz'
    id = Column(Integer, primary_key=True, autoincrement=True)
    image_name = Column(String, nullable=False)
    values = Column(JsonType, nullable=True)  # Array of floats

# Create
def create_uv_xyz(db: Session, point_list: UV_XYZ):
    db.add(point_list)
    db.commit()
    db.refresh(point_list)
    return point_list
    
# Read
def get_uv_xyz(db: Session, image_name: str)->List[float]:
    print("get_uv_xyz",image_name)
    list_points= db.query(UV_XYZ).filter(and_(UV_XYZ.image_name == image_name)).first()
    return list_points.values if list_points else None



