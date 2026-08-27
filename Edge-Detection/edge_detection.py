import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Get project folder
project_folder = Path(__file__).resolve().parent.parent

# Path to input image
image_path = project_folder / "images" / "Lenna.png"

# Read image in grayscale
image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

# Check if image loaded
if image is None:
    print("Error: Could not load image.")
    exit()

# ------------------------------------------------
# A) PREWITT FILTER
# ------------------------------------------------

# Horizontal kernel
prewitt_x = np.array([
    [-1, 0, 1],
    [-1, 0, 1],
    [-1, 0, 1]
])

# Vertical kernel
prewitt_y = np.array([
    [-1, -1, -1],
    [ 0,  0,  0],
    [ 1,  1,  1]
])

# Apply kernels
px = cv2.filter2D(image, cv2.CV_64F, prewitt_x)
py = cv2.filter2D(image, cv2.CV_64F, prewitt_y)

# Calculate gradient magnitude
prewitt = np.sqrt(px ** 2 + py ** 2)

# Convert to 8-bit
prewitt = cv2.convertScaleAbs(prewitt)


# ------------------------------------------------
# B) SOBEL FILTER
# ------------------------------------------------

# Horizontal gradient
sx = cv2.Sobel(
    image,
    cv2.CV_64F,
    1,
    0,
    ksize=3
)

# Vertical gradient
sy = cv2.Sobel(
    image,
    cv2.CV_64F,
    0,
    1,
    ksize=3
)

# Calculate gradient magnitude
sobel = np.sqrt(sx ** 2 + sy ** 2)

# Convert to 8-bit
sobel = cv2.convertScaleAbs(sobel)


# ------------------------------------------------
# C) LAPLACIAN FILTER
# ------------------------------------------------

laplacian = cv2.Laplacian(
    image,
    cv2.CV_64F
)

# Convert to 8-bit
laplacian = cv2.convertScaleAbs(laplacian)


# ------------------------------------------------
# Display Results
# ------------------------------------------------

plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.imshow(image, cmap="gray")
plt.title("Original Grayscale Image")
plt.axis("off")

plt.subplot(2, 2, 2)
plt.imshow(prewitt, cmap="gray")
plt.title("Prewitt")
plt.axis("off")

plt.subplot(2, 2, 3)
plt.imshow(sobel, cmap="gray")
plt.title("Sobel")
plt.axis("off")

plt.subplot(2, 2, 4)
plt.imshow(laplacian, cmap="gray")
plt.title("Laplacian")
plt.axis("off")

plt.tight_layout()
plt.show()