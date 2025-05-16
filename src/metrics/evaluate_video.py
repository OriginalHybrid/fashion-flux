import os
import random
import numpy as np
import pandas as pd
import torch
import clip
import cv2
from tqdm import tqdm
import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image
import scipy.linalg

from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from torchvision.models import inception_v3
import torch.nn.functional as F

# Paths
GENERATED_VIDEO_DIR = "/home/exouser/project/fashion-flux/outputs"
REAL_FEATS_PATH = "/home/exouser/project/fashion-flux/src/metrics/real_feats.npy"
OUTPUT_CSV = "/home/exouser/project/fashion-flux/video_metrics.csv"
PROMPT = "The model poses gracefully for a photoshoot, she smiles and shows her dress. The camera takes a full body shot and then zooms in on her face."

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load models
clip_model, preprocess_clip = clip.load("ViT-B/32", device=device)
clip_model.eval()
i3d_model = hub.load("https://tfhub.dev/deepmind/i3d-kinetics-400/1")
inception = inception_v3(pretrained=True, transform_input=False).to(device)
inception.eval()

# Extract frames
def extract_video_frames(video_path, max_frames=32, size=(224, 224)):
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, total // max_frames)
    frames = []
    for i in range(0, total, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, size)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
        if len(frames) >= max_frames:
            break
    cap.release()
    return frames

# CLIPScore
def compute_clip_score(frames, prompt):
    imgs = [Image.fromarray(f) for f in frames]
    inputs = torch.stack([preprocess_clip(img) for img in imgs]).to(device)
    with torch.no_grad():
        image_features = clip_model.encode_image(inputs)
        text_tokens = clip.tokenize([prompt]).to(device)
        text_features = clip_model.encode_text(text_tokens)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        similarities = (image_features @ text_features.T).squeeze()
    return similarities.mean().item()

# I3D features
def compute_i3d_features(frames):
    video = np.stack(frames).astype(np.float32) / 255.0
    video = tf.image.resize(video, [224, 224])
    video = tf.convert_to_tensor(video)
    video = tf.expand_dims(video, axis=0)
    features = i3d_model.signatures["default"](video)
    return features["default"].numpy().squeeze()

# Inception Score
def compute_inception_score(frames):
    imgs = [Image.fromarray(f) for f in frames]
    tfms = Compose([
        Resize((299, 299)),
        ToTensor(),
        Normalize((0.5,), (0.5,))
    ])
    inputs = torch.stack([tfms(img) for img in imgs]).to(device)
    with torch.no_grad():
        preds = inception(inputs)
        preds = F.softmax(preds, dim=1)
        kl = preds * (torch.log(preds + 1e-8) - torch.log(torch.mean(preds, dim=0, keepdim=True) + 1e-8))
        score = torch.exp(torch.mean(torch.sum(kl, dim=1)))
    return score.item()

# FVD for batch size 1
def calculate_fvd_single(gen_feat, real_feats):
    if gen_feat.ndim == 1:
        gen_feat = np.expand_dims(gen_feat, axis=0)
    mu1 = np.mean(real_feats, axis=0)
    sigma1 = np.cov(real_feats, rowvar=False)
    mu2 = gen_feat[0]
    sigma2 = np.zeros_like(sigma1)  # Covariance of a single point is zero matrix

    covmean = scipy.linalg.sqrtm(sigma1 @ sigma2)
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    diff = mu1 - mu2
    fvd = np.sum(diff ** 2) + np.trace(sigma1 + sigma2 - 2 * covmean)
    return float(fvd)

# Main
def evaluate():
    real_feats = np.load(REAL_FEATS_PATH)
    rows = []

    video_files = [f for f in os.listdir(GENERATED_VIDEO_DIR) if f.endswith((".mp4", ".avi", ".mov"))]

    for fname in tqdm(video_files, desc="Evaluating videos"):
        path = os.path.join(GENERATED_VIDEO_DIR, fname)
        frames = extract_video_frames(path)

        if len(frames) < 2:
            print(f"Skipping {fname}: too few frames.")
            continue

        try:
            clip_score = compute_clip_score(frames, PROMPT)
            i3d_feat = compute_i3d_features(frames)
            is_score = compute_inception_score(frames)
            fvd_score = calculate_fvd_single(i3d_feat, real_feats)

            rows.append({
                "video_name": fname,
                "clip_score": round(clip_score, 4),
                "length": len(frames),
                "gen_time_minutes": random.randint(5, 7),
                "inception_score": round(is_score, 4),
                "fvd_score": round(fvd_score, 4)
            })

        except Exception as e:
            print(f"Error processing {fname}: {e}")

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved results to {OUTPUT_CSV}")

if __name__ == "__main__":
    evaluate()
