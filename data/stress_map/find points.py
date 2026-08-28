import matplotlib.pyplot as plt
from PIL import Image

# Load your red-dot image
img_path = "Irr Region of Interest Marked.png"
img = Image.open(img_path)

# Show image and collect points
fig, ax = plt.subplots()
ax.imshow(img)
ax.set_title("Click the 12 Raman Red Dots (left to right, top to bottom)")
ax.axis('on')

# Use ginput to manually click 12 points
clicked_points = plt.ginput(12, timeout=0)  # wait for 15 clicks, no timeout
plt.close()

# Convert to NumPy array
import numpy as np
coords = np.array(clicked_points)

# Save or print the coordinates
print("Clicked coordinates (in image pixels):")
for i, (x, y) in enumerate(coords):
    print(f"Point {i+1}: ({x:.1f}, {y:.1f})")

# Optionally: save to a file
np.savetxt("stress_map_points_pixels.csv", coords, delimiter=",", header="x,y", comments='')
