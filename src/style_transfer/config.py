import argparse
import os

def parse_args():
    args = argparse.Namespace()
    args.base_model_path = "booksforcharlie/stable-diffusion-inpainting"
    args.resume_path = "zhengchong/CatVTON"
    args.output_dir = "resource/demo/output"
    args.width = 768
    args.height = 1024
    args.repaint = True
    args.allow_tf32 = True
    args.mixed_precision = "bf16"
    args.cloth_type = "upper"
    args.num_inference_steps = 50
    args.guidance_scale = 2.5
    args.seed = 42
    return args