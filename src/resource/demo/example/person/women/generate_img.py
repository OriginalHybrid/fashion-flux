import json
import numpy as np
import cv2

# Load the LabelMe JSON file
with open("1-model_3.json") as f:
    data = json.load(f)

# Get image shape (LabelMe exports images as 2048x1024 by default unless changed)
height = 1024
width = 768

# Create a black image
mask = np.zeros((height, width), dtype=np.uint8)

# Get the lehenga polygon points
for shape in data['shapes']:
    if shape['label'] == 'lehenga':
        points = np.array(shape['points'], dtype=np.int32)
        cv2.fillPoly(mask, [points], 255)  # Fill with white

# Save or display the mask
cv2.imwrite("lehenga_mask.png", mask)
cv2.imshow("Lehenga Mask", mask)
cv2.waitKey(0)
cv2.destroyAllWindows()
