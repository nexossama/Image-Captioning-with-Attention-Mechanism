
import io
import logging
import os
import pickle
from typing import List
from PIL import Image
from fastapi import UploadFile
from helpers.config import get_settings
from models.ImageCaptioning import EncoderDecoder
import dill as pickle
import pandas as pd

logger = logging.getLogger('uvicorn.error')

class GenerateController:
    
    def __init__(self):
        self.app_settings = get_settings()
    
    def generate_caption(self,image):
        src_path=os.path.dirname(os.path.dirname(__file__))
        model_path=os.path.join(src_path,"assets","attention_model_state_200.pth")
        print(model_path)
        model = EncoderDecoder.load_model(model_path)
        
        train_dataset_path=os.path.join(src_path,"assets","train_dataset (1).pkl")
        with open(train_dataset_path,"rb")as f:
            train_dataset=pickle.load(f)
        model.train_dataset=train_dataset
        
        caption, _, _ = model.get_caps_from_image(image)
        
        return caption
    

    async def create_story(self, files: List[UploadFile]) -> List[str]:
        captions = []
        for file in files:
            image_data = await file.read()
            # print(image_data)
            img = Image.open(io.BytesIO(image_data)).convert("RGB")
            caption = self.generate_caption(img)
            captions.append(caption)
        return captions
