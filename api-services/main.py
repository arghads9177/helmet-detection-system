"""api-services entry point. Thin — routes live under app/routes/, business logic
under app/services/, which calls into ml.inference for predictions.
"""
from fastapi import FastAPI

from app.routes import health
from app.utils.config import settings

app = FastAPI(title="CV Inference API")
app.include_router(health.router)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host=settings.api_host, port=settings.api_port)
