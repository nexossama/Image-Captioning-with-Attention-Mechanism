import io
import logging
from typing import List
from PIL import Image
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from controllers import GenerateController

logger = logging.getLogger("uvicorn.error")

generate_story_router = APIRouter(
    prefix="/api/v1/generate",
    tags=["api_v1","generate_story","generate_caption"]
)

@generate_story_router.post("/story")
async def generate_story(files: List[UploadFile] = File(...),story_type: str = Form(...)):
    print("hello")
    generator=GenerateController()

    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")

    try:
        captions,story = await generator.create_story(files,story_type)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {"captions":captions,
            "story": story}
