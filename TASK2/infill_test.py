import cv2
import numpy as np
from matplotlib import pyplot as plt

def inpaint_caption(image_path, mask_path):
    # Read the image and the mask
    image = cv2.imread(image_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    # Apply inpainting
    inpainted_image = cv2.inpaint(image, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)

    # Display the original and inpainted images side by side
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')

    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(inpainted_image, cv2.COLOR_BGR2RGB))
    plt.title('Inpainted Image')

    plt.show()

# Example usage
image_path = '../hateful_memes/img/01295.png'
mask_path = 'path/to/your/mask.png'
inpaint_caption(image_path, mask_path)
