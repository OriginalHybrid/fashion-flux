import os
import cv2
import random
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from tqdm import tqdm

# --- Config ---
REAL_VIDEO_DIR = "/home/exouser/project/fashion-flux/src/metrics/UCF-101"
FEAT_PATH = "/home/exouser/project/fashion-flux/src/metrics/real_feats.npy"
MAX_VIDEOS = 5000  # Adjust as needed

# --- Load I3D model from TensorFlow Hub ---
print("🔄 Loading I3D model from TensorFlow Hub...")
i3d_model = hub.load("https://tfhub.dev/deepmind/i3d-kinetics-400/1")
print("✅ I3D model loaded.")

# --- Frame extraction ---
def extract_frames(video_path, max_frames=32, size=(224, 224)):
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

# --- I3D Feature Extraction ---
def compute_i3d_features(frames):
    video = np.stack(frames).astype(np.float32) / 255.0
    video = tf.image.resize(video, [224, 224])
    video = tf.convert_to_tensor(video)
    video = tf.expand_dims(video, axis=0)  # (1, T, H, W, C)

    # ✅ Correct usage with signature
    features = i3d_model.signatures["default"](video)
    return features["default"].numpy().squeeze()

# --- Main Feature Generation ---
def generate_real_features(real_video_dir, output_path):
    feats = []
    all_files = []

    # Collect all .avi files recursively
    for root, _, files in os.walk(real_video_dir):
        for f in files:
            if f.endswith(".avi"):
                all_files.append(os.path.join(root, f))

    if not all_files:
        raise RuntimeError("❌ No .avi files found in the specified directory.")

    selected = random.sample(all_files, min(MAX_VIDEOS, len(all_files)))
    print(f"🧠 Processing {len(selected)} real videos for I3D features...")

    for path in tqdm(selected, desc="🔍 Extracting I3D features"):
        frames = extract_frames(path)
        if len(frames) < 2:
            print(f"⚠️ Skipping {path} — too few frames.")
            continue
        try:
            feat = compute_i3d_features(frames)
            feats.append(feat)
        except Exception as e:
            print(f"❌ Failed on {os.path.basename(path)}: {e}")

    if len(feats) == 0:
        raise RuntimeError("❌ All video feature extractions failed.")

    feats = np.stack(feats)
    np.save(output_path, feats)
    print(f"\n✅ Saved I3D features to: {output_path}")

# --- Run ---
if __name__ == "__main__":
    generate_real_features(REAL_VIDEO_DIR, FEAT_PATH)
