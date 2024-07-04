import datetime
from fastapi import HTTPException, Depends, Security
from fastapi.security import HTTPBearer, OAuth2PasswordBearer
from jose import jwt, JWTError
from typing import Optional

from pydantic import BaseModel
from python_server.utils.errors import BAD_REQUEST_HTTPEXCEPTION
from python_server.settings.settings import settings

SECRET_KEY = settings().token.secret_key

ALGORITHM = "HS256"

security = HTTPBearer()

class UserRights(BaseModel):
    models:list[str]


def generate_token(user_rights:UserRights):
    payload = {
        'exp': datetime.datetime.utcnow() + datetime.timedelta(days=100),
        'iat': datetime.datetime.utcnow(),
        'scope': user_rights  # Custom payload data to include user rights
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def verify_token(http_authorization_credentials=Security(security)):
    try:
        token = http_authorization_credentials.credentials
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except JWTError:
        raise BAD_REQUEST_HTTPEXCEPTION("Invalid token:"+token)
    
