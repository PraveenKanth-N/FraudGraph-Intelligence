from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.app.config import settings
from backend.app.api.v1.endpoints import health, dashboard, datasets

app = FastAPI(
    title=settings.PROJECT_NAME,
    description="Memory-Aware CUDA Acceleration of Graph-Based Financial Fraud Detection Using Parallel Neighbor Aggregation",
    version=settings.VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount API Routers
app.include_router(health.router, prefix=settings.API_V1_PREFIX, tags=["Diagnostics & Telemetry"])
app.include_router(dashboard.router, prefix=settings.API_V1_PREFIX, tags=["Dashboard & Workspaces"])
app.include_router(datasets.router, prefix=settings.API_V1_PREFIX, tags=["Datasets"])

@app.get("/")
def root():
    return {
        "service": settings.PROJECT_NAME,
        "course": "VIT Vellore — CSI4007 GPU Programming",
        "status": "operational",
        "api_v1": settings.API_V1_PREFIX,
        "docs": "/docs"
    }
