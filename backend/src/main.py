# src/main.py
from fastapi import FastAPI

app = FastAPI(title="Healthcare Resource Optimization API")

@app.get("/")
def read_root():
    return {"status": "Backend is running", "phase": "Digital Twin & Optimization Skeleton"}