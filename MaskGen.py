import boto3
import cv2
import os
import numpy as np
from PIL import Image, ImageDraw

def mask_generator(path, region_name):
    # Initialize the Textract client
    client = boto3.client('textract', region_name=region_name)

    # Open the image
    with open(path, 'rb') as ima:
        image = Image.open(ima).convert('RGB')
        mask = image.copy()  # Create a copy for the mask
        pixels = image.load()
        pixels_msk = mask.load()

        # Reset the file pointer and read the image for Textract
        ima.seek(0)
        response = client.detect_document_text(Document={'Bytes': ima.read()})

    # Get the text blocks from Textract response
    blocks = response['Blocks']
    width, height = image.size

    # Initialize the mask with black color
    for m in range(width):
        for n in range(height):
            pixels_msk[m, n] = (0, 0, 0)

    # Process each block detected by Textract
    for block in blocks:
        if block['BlockType'] == "LINE":
            points = [(width * polygon['X'], height * polygon['Y']) for polygon in block['Geometry']['Polygon']]

            # Calculate bounding box coordinates
            minXL = max(0, int(min(points[0][0], points[3][0])))
            maxXR = min(width, int(max(points[1][0], points[2][0])))
            minYL = max(0, int(min(points[0][1], points[1][1])))
            maxYU = min(height, int(max(points[2][1], points[3][1])))

            # Fill the bounding box area with white color in both image and mask
            for i in range(minXL, maxXR):
                for j in range(minYL, maxYU):
                    pixels[i, j] = (255, 255, 255)
                    pixels_msk[i, j] = (255, 255, 255)

    # Convert images to numpy arrays for saving with OpenCV
    image = np.array(image)
    mask = np.array(mask)

    # Resize images if needed
    resized_mask = cv2.resize(mask, (720, 512), interpolation=cv2.INTER_AREA)
    resized_image = cv2.resize(image, (720, 512), interpolation=cv2.INTER_AREA)

    # Ensure directories exist
    os.makedirs('./Masks', exist_ok=True)
    os.makedirs('./Images', exist_ok=True)

    # Save the mask and final cut image
    cv2.imwrite('./Masks/mask.jpg', cv2.cvtColor(resized_mask, cv2.COLOR_RGB2BGR))
    cv2.imwrite('./Images/image.jpg', cv2.cvtColor(resized_image, cv2.COLOR_RGB2BGR))

    print('Mask and Image are generated')
    print(f'Number of text blocks: {len(blocks)}')
