from fastapi import Depends,HTTPException,status
from fastapi.security import OAuth2PasswordBearer
from app.services import verify_access_token
from app.database import users_collect

oauth2_scheme=OAuth2PasswordBearer(tokenUrl="/app/login")
def get_current_user(token:str=Depends(oauth2_scheme)):
    payload=verify_access_token(token)
    if payload is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,detail="Invalid or expired token")
    email=payload.get("sub")
    if email is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,detail="Invalid token")
    user=users_collect.find_one({"email":email})
    if user is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,detail="User not found")
    return user