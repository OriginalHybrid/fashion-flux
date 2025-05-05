import argparse
import os
from datetime import datetime

import torch
from diffusers.image_processor import VaeImageProcessor
from huggingface_hub import snapshot_download
from PIL import Image

from style_transfer.model.cloth_masker import AutoMasker, vis_mask
from style_transfer.model.pipeline import CatVTONPipeline
from style_transfer.model.flux.pipeline_flux_tryon import FluxTryOnPipeline
from style_transfer.utils import init_weight_dtype, resize_and_crop, resize_and_padding
from style_transfer.config import parse_args

args = parse_args()

def image_grid(imgs, rows, cols):
    assert len(imgs) == rows * cols

    w, h = imgs[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid

repo_path = snapshot_download(repo_id=args.resume_path)

# Pipeline
pipeline = CatVTONPipeline(
    base_ckpt=args.base_model_path,
    attn_ckpt=repo_path,
    attn_ckpt_version="mix",
    weight_dtype=init_weight_dtype(args.mixed_precision),
    use_tf32=args.allow_tf32,
    device='cuda'
)

# AutoMasker
mask_processor = VaeImageProcessor(vae_scale_factor=8, do_normalize=False, do_binarize=True, do_convert_grayscale=True)
automasker = AutoMasker(
    densepose_ckpt=os.path.join(repo_path, "DensePose"),
    schp_ckpt=os.path.join(repo_path, "SCHP"),
    device='cuda', 
)

def submit_function(
    person_image,
    cloth_image,
    cloth_type,
    num_inference_steps,
    guidance_scale,
    seed
):
    # Set random seed
    tmp_folder = args.output_dir
    date_str = datetime.now().strftime("%Y%m%d%H%M%S")
    result_save_path = os.path.join(tmp_folder, date_str[:8], date_str[8:] + ".png")
    if not os.path.exists(os.path.join(tmp_folder, date_str[:8])):
        os.makedirs(os.path.join(tmp_folder, date_str[:8]))

    generator = None
    if seed != -1:
        generator = torch.Generator(device='cuda').manual_seed(seed)

    # person_image = Image.open(person_image).convert("RGB")
    # cloth_image = Image.open(cloth_image).convert("RGB")
    person_image = resize_and_crop(person_image, (args.width, args.height))
    cloth_image = resize_and_padding(cloth_image, (args.width, args.height))
    
    # Process mask
    mask = automasker(
        person_image,
        cloth_type
    )['mask']
    mask = mask_processor.blur(mask, blur_factor=9)

    # Inference
    # try:
    result_image = pipeline(
        image=person_image,
        condition_image=cloth_image,
        mask=mask,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=generator
    )[0]
    
    masked_person = vis_mask(person_image, mask)
    save_result_image = image_grid([person_image, masked_person, cloth_image, result_image], 1, 4)
    save_result_image.save(result_save_path)
    return masked_person, result_image


def person_example_fn(image_path):
    return image_path


    result = simple_vton_inference(
        person_image_path="resource/demo/example/person/men/model_5.png",
        cloth_image_path="resource/demo/example/condition/upper/21514384_52353349_1000.jpg",
        cloth_type="upper",
        num_inference_steps=5,
        guidance_scale=2.5,
        seed=42,
        show_type="input & mask & result",
        model="sd"
    )
    # result.show()