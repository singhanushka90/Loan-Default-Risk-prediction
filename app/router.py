from fastapi import APIRouter
from app.schemas import SignUp,PredictionRequest
from app.database import users_collect
from fastapi import HTTPException,Depends
from app.services import hash_password , verify_password , create_access_token
from fastapi.security import OAuth2PasswordRequestForm,OAuth2PasswordBearer
from app.dependencies import get_current_user
from app.model_service import predict
import pandas as pd


router=APIRouter()
oauth2_scheme=OAuth2PasswordBearer(tokenUrl="/app/login")
@router.get('/home')
def home():
    return {"message":"Loan Detection API running"}


@router.post('/signup')
def signup(user:SignUp):
    existing_user=users_collect.find_one({"email":user.email})
    if existing_user:
        raise HTTPException(status_code=400,detail="Email already exist")
    hashed_password=hash_password(user.password)
    users_collect.insert_one({
        "username":user.username,
        "email":user.email,
        "password":hashed_password
    })
    return {"message":"User registered successfully"}

@router.post('/login')
def login(form_data:OAuth2PasswordRequestForm=Depends()):
    db_user=users_collect.find_one({"email":form_data.username})
    if not db_user:
        raise HTTPException(status_code=401,detail="Invalid Email or Password")
    if not verify_password(form_data.password,db_user['password']):
        raise HTTPException(status_code=401,detail="Invalid Email or Password")
    token=create_access_token(
        data={"sub":db_user["email"],
              "username":db_user["username"],
            }
    )
    return {
        "access_token":token,
        "token_type":"bearer"
    }
@router.get("/profile")
def profile(current_user=Depends(get_current_user)):
    return {
        "username":current_user["username"],
        "email":current_user["email"]
    }

@router.get("/profile")
def profile(current_user=Depends(get_current_user)):
    return {"message":"Profile accessed successfully","user":current_user}


@router.post("/predict")
def make_prediction(data:PredictionRequest,current_user=Depends(get_current_user)):
    input_data=pd.DataFrame([data.model_dump()])
    prediction,probability=predict(input_data)
    return {"prediction":int(prediction),"probability":float(probability)}