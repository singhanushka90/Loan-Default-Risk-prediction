from pydantic import BaseModel , EmailStr

class SignUp(BaseModel):
    username:str
    email:EmailStr
    password:str

class TokenResponse(BaseModel):
    access_token:str
    token_type:str

class UserResponse(BaseModel):
    username:str
    email:str


class PredictionRequest(BaseModel):
    
    CODE_GENDER:str
    NAME_INCOME_TYPE:str
    NAME_EDUCATION_TYPE:str
    NAME_FAMILY_STATUS:str
    NAME_HOUSING_TYPE:str
    OCCUPATION_TYPE:str
    FLAG_OWN_CAR:str
    FLAG_OWN_REALTY:str
    ORGANIZATION_TYPE:str
    CNT_CHILDREN:int
    CNT_FAM_MEMBERS:float
    AMT_INCOME_TOTAL:float
    AMT_CREDIT:float
    AMT_ANNUITY:float
    AMT_GOODS_PRICE:float
    DAYS_BIRTH:int
    DAYS_EMPLOYED:int
    REGION_RATING_CLIENT:int
    REGION_RATING_CLIENT_W_CITY:int
    REGION_POPULATION_RELATIVE:float
    EXT_SOURCE_1:float
    EXT_SOURCE_2:float
    EXT_SOURCE_3:float
    DAYS_LAST_PHONE_CHANGE:float