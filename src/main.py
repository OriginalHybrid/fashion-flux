import os
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import FileResponse, JSONResponse
import numpy as np
import io
from PIL import Image
import base64
from style_transfer.inference import submit_function
from style_transfer.config import parse_args
import json
from datetime import datetime

import sys

sys.path.append(os.path.join(os.path.dirname(__file__), 'style_transfer'))

OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)


app = FastAPI()
# Dummy generation function

@app.post("/style_transfer")
async def style_transfer(
    person_image: UploadFile = File(...),
    cloth_image: UploadFile = File(...),
    config_json: str = Form(...)
):
    # Step 1: Read both uploaded images
    person_bytes = await person_image.read()
    cloth_bytes = await cloth_image.read()
    config_dict = json.loads(config_json)
    args = parse_args()

    person_img = Image.open(io.BytesIO(person_bytes)).convert("RGB")
    cloth_img = Image.open(io.BytesIO(cloth_bytes)).convert("RGB")
    cloth_type = config_dict.get('cloth_type', args.cloth_type)
    num_inference_steps = config_dict.get('num_inference_steps', args.num_inference_steps)
    guidance_scale =  config_dict.get('guidance_scale', args.guidance_scale)
    seed =  config_dict.get('seed', args.seed)
                 
    # Step 2: Run the generation model (placeholder)
    masked_img, result_img = submit_function(
                        person_img,
                        cloth_img,
                        cloth_type,
                        num_inference_steps,
                        guidance_scale,
                        seed
                    )

    def encode_image(image: Image.Image) -> str:
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    masked_path = os.path.join(OUTPUT_DIR, f"masked_{timestamp}.png")
    result_path = os.path.join(OUTPUT_DIR, f"result_{timestamp}.png")
    masked_img.save(masked_path)
    result_img.save(result_path)

    # Step 5: Encode and return
    return JSONResponse(content={
        "masked_image": encode_image(masked_img),
        "result_image": encode_image(result_img)
    })

@app.post("/pose_transfer")
async def pose_transfer(
    person_image: UploadFile = File(...),
    cloth_image: UploadFile = File(...),
    config_json: str = Form(...)
):
    pass