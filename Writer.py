import cv2

def writer(image, text, coord):
    if len(text) != len(coord):
        raise ValueError("The length of 'text' and 'coord' must be the same.")
    
    # Font parameters
    font = cv2.FONT_HERSHEY_COMPLEX_SMALL
    font_scale = 1
    color = (255, 255, 255)  # White
    thickness = 1
    
    for i, crd in enumerate(coord):
        try:
            org = crd
            image = cv2.putText(image, text[i], org, font, 
                                font_scale, color, thickness, cv2.LINE_AA)
        except Exception as e:
            print(f"An error occurred while adding text: {e}")

    print('New image generated')
    return image
