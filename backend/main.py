import os
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from dotenv import load_dotenv

from routers import predict, history, stats, analytics, chat, profile
from schemas.prediction import ErrorResponse

# Load environment variables
load_dotenv()

# Initialize FastAPI App
app = FastAPI(
    title=os.getenv("APP_NAME", "YouTube Prediction API"),
    description="Backend API for predicting YouTube video performance and diagnosing view drops using Machine Learning.",
    version=os.getenv("APP_VERSION", "1.0.0"),
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS Configuration
allowed_origins = os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin.strip() for origin in allowed_origins if origin.strip()],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register Routers
app.include_router(predict.router)
app.include_router(history.router)
app.include_router(stats.router)
app.include_router(analytics.router)
app.include_router(chat.router)
app.include_router(profile.router)

# Health Check Endpoint
@app.get("/health", tags=["Health"])
async def health_check():
    return {"status": "ok"}

# Global Exception Handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error="HTTP Exception",
            detail=exc.detail
        ).model_dump()
    )

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content=ErrorResponse(
            error="Validation Error",
            detail=str(exc.errors())
        ).model_dump()
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=os.getenv("DEBUG", "False").lower() == "true")
