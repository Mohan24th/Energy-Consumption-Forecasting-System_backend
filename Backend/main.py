# main.py (inside Backend folder)
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from model_loader import load_model
from predict import router as predict_router
from info import router as info_router

app = FastAPI(
    title="Energy Consumption Forecasting API",
    description="Predict energy consumption for homes and commercial buildings",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers – FIXED
app.include_router(predict_router)
app.include_router(info_router)

@app.on_event("startup")
async def startup_event():
    """Load ML model when the application starts."""
    load_model()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)   # FIXED