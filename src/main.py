from fastapi import FastAPI
from routes import base , generate_story
from fastapi.middleware.cors import CORSMiddleware
from helpers.config import get_settings

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)
app.include_router(base.base_router)
app.include_router(generate_story.generate_story_router)