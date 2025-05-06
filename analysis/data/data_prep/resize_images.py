import os
from PIL import Image
import numpy as np

# Set input and output directories
input_folder = 'your_input_folder'   # <-- Replace this
output_folder = 'resized_images'
os.makedirs(output_folder, exist_ok=True)

# Step 1: Get list of images
image_files = sorted([
    f for f in os.listdir(input_folder)
    if f.lower().endswith(('.jpg', '.jpeg', '.png'))
])

# Step 2: Rename and collect dimensions
widths, heights = [], []

for idx, filename in enumerate(image_files, start=1):
    old_path = os.path.join(input_folder, filename)

    with Image.open(old_path) as img:
        widths.append(img.width)
        heights.append(img.height)

        # Convert to RGB to avoid issues with PNG
        img = img.convert("RGB")

        new_filename = f"images_{idx:02d}.png"
        new_path = os.path.join(input_folder, new_filename)
        img.save(new_path, format="PNG")

    # Optionally remove the original file
    if filename != new_filename:
        os.remove(old_path)

# Step 3: Compute average dimensions
avg_width = int(np.mean(widths))
avg_height = int(np.mean(heights))
print(f"Average size: {avg_width} x {avg_height}")

# Step 4: Resize and save to new folder
renamed_files = sorted([
    f for f in os.listdir(input_folder)
    if f.startswith("images_") and f.endswith(".png")
])

for filename in renamed_files:
    path = os.path.join(input_folder, filename)
    with Image.open(path) as img:
        resized_img = img.resize((avg_width, avg_height))
        resized_img.save(os.path.join(output_folder, filename), format="PNG")

print(f"✅ All images resized to {avg_width}x{avg_height} and saved as PNGs in '{output_folder}'")