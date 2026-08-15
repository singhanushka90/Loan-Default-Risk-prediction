from fastapi import FastAPI
from app.router import router

app=FastAPI(title="Loan Detection API",version="1.0.0")

app.include_router(router,prefix="/app")
