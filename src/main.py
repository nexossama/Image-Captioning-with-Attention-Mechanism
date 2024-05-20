from fastapi import FastAPI
from .routes import base , generate_story
from .helpers.config import get_settings

app = FastAPI()
app.include_router(base.base_router)
app.include_router(generate_story.generate_story_router)