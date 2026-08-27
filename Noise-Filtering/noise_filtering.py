import cv2
import matplotlib.pyplot as plt
from pathlib import Path

# Get project folder
project_folder = Path(__file__).resolve().parent.parent

# Path to input image
image_path = project_folder / "images" / "Lenna.png"

# Read image
image = cv2.imread(str(image_path))

# Check if image loaded
if image is None:
    print("Error: Could not load image.")
    exit()

# Convert BGR to RGB for matplotlib
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# ------------------------------------------------
# A) Average Filter
# ------------------------------------------------

average = cv2.blur(image, (5, 5))
average_rgb = cv2.cvtColor(average, cv2.COLOR_BGR2RGB)

# ------------------------------------------------
# B) Gaussian Filter
# ------------------------------------------------

gaussian = cv2.GaussianBlur(image, (5, 5), 0)
gaussian_rgb = cv2.cvtColor(gaussian, cv2.COLOR_BGR2RGB)

# ------------------------------------------------
# C) Median Filter
# ------------------------------------------------

median = cv2.medianBlur(image, 5)
median_rgb = cv2.cvtColor(median, cv2.COLOR_BGR2RGB)

# ------------------------------------------------
# Display Results
# ------------------------------------------------

plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.imshow(image_rgb)
plt.title("Original Image")
plt.axis("off")

plt.subplot(2, 2, 2)
plt.imshow(average_rgb)
plt.title("Average Filter")
plt.axis("off")

plt.subplot(2, 2, 3)
plt.imshow(gaussian_rgb)
plt.title("Gaussian Filter")
plt.axis("off")

plt.subplot(2, 2, 4)
plt.imshow(median_rgb)
plt.title("Median Filter")
plt.axis("off")

plt.tight_layout()
plt.show()