import cv2
import numpy as np
import os

def unsharp_mask(image, kernel_size=(5, 5), sigma=1.0, amount=1.0, threshold=0):
    """Return a sharpened version of the image, using an unsharp mask."""
    blurred = cv2.GaussianBlur(image, kernel_size, sigma)
    sharpened = float(amount + 1) * image - float(amount) * blurred
    sharpened = np.clip(sharpened, 0, 255).round().astype(np.uint8)

    if threshold > 0:
        low_contrast_mask = np.absolute(image - blurred) < threshold
        np.copyto(sharpened, image, where=low_contrast_mask)
    
    return sharpened

def postprocess():
    # Ensure the results directory exists
    if not os.path.exists('./results'):
        raise FileNotFoundError("The directory './results' does not exist.")

    # Load the image
    image_path = './results/image.jpg'
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"The image file '{image_path}' does not exist.")
    
    im = cv2.imread(image_path)
    if im is None:
        raise ValueError(f"Failed to read the image '{image_path}'.")

    # Apply median blur
    final = cv2.medianBlur(im, 3)
    
    # Apply unsharp masking
    sharpened_image = unsharp_mask(final)
    
    # Save the processed image
    output_path = './results/sharpened_image.jpg'
    cv2.imwrite(output_path, sharpened_image)
    print(f"Processed image saved as '{output_path}'")

