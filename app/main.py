from fastapi import FastAPI
from app.routes import analyze
from app.database.db import Base, engine

app = FastAPI()

# create table
Base.metadata.create_all(bind=engine)

app.include_router(analyze.router)