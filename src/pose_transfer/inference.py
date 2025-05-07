from typing import Any, Dict
import numpy as np
import torch
import torch.nn as nn
from pose_transfer.pipeline import LeffaPipeline

class LeffaInference:
    def __init__(self, model: nn.Module) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = model.to(self.device)
        self.model.eval()
        self.pipe = LeffaPipeline(model=self.model)

    def to_gpu(self, data: Dict[str, Any]) -> Dict[str, Any]:
        for k, v in data.items():
            if isinstance(v, torch.Tensor):
                data[k] = v.to(self.device)
        return data

    def __call__(self, data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        data = self.to_gpu(data)
        images = self.pipe(
            src_image=data["src_image"],
            ref_image=data["ref_image"],
            mask=data["mask"],
            densepose=data["densepose"],
            num_inference_steps=kwargs.get("num_inference_steps", 50),
            guidance_scale=kwargs.get("guidance_scale", 2.5),
            generator=torch.Generator(self.pipe.device).manual_seed(kwargs.get("seed", 42)),
        )[0]
        return {"generated_image": images}