import sys
import os
import numpy as np
from PIL import Image  # Pillow library for python image processing
from huggingface_hub import snapshot_download
import torch
from IPython.display import display
import gradio as gr

# imported modules from the .py files. 
from transform import LeffaTransform        
from model import LeffaModel
from inference import LeffaInference
from densepose_predictor import DensePosePredictor  
#from leffa_utils.densepose_predictor import DensePosePredictor



# 1) Ensure ckpts are present
if not os.path.isdir("./ckpts"):
    snapshot_download(repo_id="franciszzj/Leffa", local_dir="./ckpts")
else :
    print("ckpts folder already exists. Skipping download.")


# uses code from the model.py file

# 3) Load the pose-transfer model & inference engine
pt_model = LeffaModel(
    pretrained_model_name_or_path="./ckpts/stable-diffusion-xl-1.0-inpainting-0.1",
    #pretrained_model="./ckpts/pose_transfer.pth",   # This file holds all the fine-tuned parameter updates on top of the base model.                        # 19GB model size
    pretrained_model="./ckpts/pose_transfer_fp16.pth",     # 16 but Quantized model Size reduced to 9.7GB
    dtype="float16",
)



# uses code from the inference.py file

pt_inference = LeffaInference(model=pt_model)



# uses code from the transform.py file
#transform = LeffaTransform()
transform = LeffaTransform(dataset="pose_transfer")  # added dataset="pose_transfer" to the function call


# 2) Initialize the DensePose predictor
densepose_predictor = DensePosePredictor(
    config_path="./ckpts/densepose/densepose_rcnn_R_50_FPN_s1x.yaml",   # config file 
    weights_path="./ckpts/densepose/model_final_162be9.pkl",            # model weights in pickle format
)




def pose_transfer(
    #src_image_path: str,   # target_pose image
    #ref_image_path: str,   # ref person image 
    src_img: Image.Image,   # added for gradio call using fastapi 
    ref_img: Image.Image,   # added for gradio call using fastapi
    steps: int = 80,        # 30 to 80
    scale: float = 3.5,
    seed: int = 42,
) -> Image.Image:
    
    # a) Load & resize both images to 768×1024
    #src = Image.open(src_image_path).convert("RGB").resize((768, 1024))     # src image is target-pose image.
    #ref = Image.open(ref_image_path).convert("RGB").resize((768, 1024))     # ref image is person image.

    src = src_img.resize((768, 1024))     # src image is target-pose image.
    ref = ref_img.resize((768, 1024))     # ref image is person image.



    # b) Create a white mask (no occlusion)
    arr = np.array(src)
    mask = Image.fromarray(np.ones_like(arr) * 255)




    # c) Compute DensePose IUV map for the SRC (target pose)
    src_arr = np.array(src)
    iuv = densepose_predictor.predict_iuv(src_arr)[:, :, ::-1]     
    #iuv = predict_iuv(src_arr)[:, :, ::-1]
    densepose = Image.fromarray(iuv)



    # d) Package into the format LeffaTransform expects
    batch = {
        "src_image": [src],
        "ref_image": [ref],
        "mask": [mask],
        "densepose": [densepose],
    }



    batch = transform(batch)



    # e) Run the diffusion model
    out = pt_inference(
        batch,
        num_inference_steps=steps,
        guidance_scale=scale,
        seed=seed,
    )

# This Invokes your LeffaInference wrapper, which under the hood:
# Moves all tensors to CUDA. 
# Calls LeffaPipeline, running:
    # VAE encoding of masked & reference images
    # Denoising loop of steps iterations with scale guidance
    # VAE decoding → a list of PIL images
    # Returns generated_image.
    
    # the first (and only) generated image
    gen = out["generated_image"][0]
    return gen


    
""" 
# Check if the script is being run directly
if __name__ == "__main__":

    person_image_path = "1.jpg"
    target_pose_image_path = "9.jpg"

    # Direct call
    output_image = pose_transfer(target_pose_image_path, person_image_path)

    #output_image.resize((300, 500))
    # Display the output image
    #display(output_image)
    # Save the output image
    output_image.save("generated_pose_transfer1.jpg")
"""


"""
# wrap so that first UI input (person) becomes `ref` and second (pose) becomes `src`
def wrapped_pose_transfer(person_image_path, target_pose_image_path, steps=80, scale=3.5, seed=42):
    # note the swap: pose_transfer expects (pose, person)
    return pose_transfer(
        src_image_path=target_pose_image_path,
        ref_image_path=person_image_path,
        steps=steps,
        scale=scale,
        seed=seed,
    )



demo = gr.Interface( 
    fn=wrapped_pose_transfer,
    inputs=[
        gr.Image(type="filepath", label="Person Image"),
        gr.Image(type="filepath", label="Target Pose Image")
    ],
    outputs=gr.Image(label="Generated Pose-Transferred Image"),
    title="Leffa Pose Transfer",
)
demo.launch(share=True)

"""