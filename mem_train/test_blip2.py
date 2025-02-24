from typing import Any, Optional
import os

import numpy as np
import torch
from PIL import Image
from lavis.models import load_model_and_preprocess

# Set the current directory as the root
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parent
# Change the working directory (optional)
os.chdir(ROOT_DIR)

device = torch.device("cuda:0") if torch.cuda.is_available() else "cpu"

if __name__ == "__main__":
    
    raw_image = Image.open("./demo_image/demo_img0.png").convert("RGB")
    txt_cmd = "Describe the image in details" # go to the window

    model, vis_processors, txt_processors = load_model_and_preprocess(
        name="blip2_feature_extractor", 
        model_type="pretrain_vitL", 
        is_eval=True, device=device)

    image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
    text_input = txt_processors["eval"](txt_cmd)
    sample = {"image": image, "text_input": [text_input]}

    # Multimodal
    features_multimodal = model.extract_features(sample)
    print(features_multimodal.multimodal_embeds.shape) # torch.Size([1, 32, 768]), 32 is the number of queries

    # Unimodal
    features_image = model.extract_features(sample, mode="image")
    features_text = model.extract_features(sample, mode="text")
    print(features_image.image_embeds.shape) # torch.Size([1, 32, 768])
    print(features_text.text_embeds.shape)   # torch.Size([1, x, 768])

    # Normalized low-dimensional unimodal features
    # low-dimensional projected features
    print(features_image.image_embeds_proj.shape) # torch.Size([1, 32, 256])
    print(features_text.text_embeds_proj.shape) # torch.Size([1, x, 256])
    similarity = (features_image.image_embeds_proj @ features_text.text_embeds_proj[:,0,:].t()).max()
    print(similarity) # tensor([[0.3642]])