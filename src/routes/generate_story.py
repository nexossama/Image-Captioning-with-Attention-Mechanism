import io
import logging
from typing import List
from PIL import Image
from fastapi import APIRouter, File, HTTPException, UploadFile
from controllers import GenerateController

logger = logging.getLogger("uvicorn.error")

generate_story_router = APIRouter(
    prefix="/api/v1/generate",
    tags=["api_v1","generate_story","generate_caption"]
)

@generate_story_router.post("/generate-story")
async def generate_story(files: List[UploadFile] = File(...)):
    print("hello")
    generator=GenerateController()

    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")

    try:
        captions = await generator.create_story(files)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {"story": " ".join(captions)}
