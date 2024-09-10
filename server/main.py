from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import util

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all HTTP methods
    allow_headers=["*"],  # Allows all headers
)

# Load saved artifacts when starting the application
@app.on_event("startup")
async def startup_event():
    util.load_saved_artifacts()

class PredictRequest(BaseModel):
    total_sqft: float
    location: str
    bhk: int
    bath: int

@app.get('/get_location_names')
async def get_location_names():
    try:
        locations = util.get_location_names()
        return JSONResponse(content={'locations': locations})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post('/predict_home_price')
async def predict_home_price(request: PredictRequest):
    try:
        estimated_price = util.get_estimated_price(
            request.location,
            request.total_sqft,
            request.bhk,
            request.bath
        )
        return JSONResponse(content={'estimated_price': estimated_price})
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    print("Starting Python FastAPI Server For Home Price Prediction...")
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=True)
