import os
import json
import numpy as np
import cv2

# Paths
json_folder = 'resized_images'
mask_folder = 'masks'

# Create the mask folder if it doesn't exist
os.makedirs(mask_folder, exist_ok=True)

# Iterate over all JSON files in the images folder
for filename in os.listdir(json_folder):
    if filename.endswith('.json'):
        json_path = os.path.join(json_folder, filename)

        # Load JSON data
        with open(json_path, 'r') as f:
            data = json.load(f)

        # Get image dimensions (update this if images have different sizes)
        height = 1173
        width = 827
        mask = np.zeros((height, width), dtype=np.uint8)

        # Draw lehenga polygon
        for shape in data['shapes']:
            if shape['label'].lower() == 'lehanga':
                points = np.array(shape['points'], dtype=np.int32)
                cv2.fillPoly(mask, [points], 255)

        # Save the mask with the same base filename but .png extension
        base_name = os.path.splitext(filename)[0]
        output_path = os.path.join(mask_folder, base_name + '.png')
        cv2.imwrite(output_path, mask)
        print(f"Saved mask: {output_path}")