from pymongo import MongoClient
from app.config import DATABASE_NAME , MONGODB_URI

client=MongoClient(MONGODB_URI)
data_b=client[DATABASE_NAME]

users_collect=data_b["users"]
