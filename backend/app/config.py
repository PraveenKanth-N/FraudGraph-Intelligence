import os
from pydantic import BaseModel
from dotenv import load_dotenv

# Load local environment if available
load_dotenv()

class Settings(BaseModel):
    PROJECT_NAME: str = os.getenv("PROJECT_NAME", "FraudGraph Intelligence")
    PROJECT_ID: str = os.getenv("PROJECT_ID", "FGI")
    VERSION: str = "1.0.0"
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development")
    DEBUG: bool = os.getenv("DEBUG", "true").lower() == "true"
    
    API_V1_PREFIX: str = os.getenv("API_V1_PREFIX", "/api/v1")
    CORS_ORIGINS: list[str] = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5174",
    ]
    
    COMPUTE_BACKEND: str = os.getenv("COMPUTE_BACKEND", "cpu_reference")
    CUDA_ARCH: str = os.getenv("CUDA_ARCH", "sm_75")
    ENABLE_GPU_FALLBACK: bool = os.getenv("ENABLE_GPU_FALLBACK", "true").lower() == "true"

settings = Settings()
