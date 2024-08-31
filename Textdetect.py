import boto3
from PIL import Image
import io

def text_extractor(bucket, document, path, region_name):
    client = boto3.client('textract', region_name=region_name)

    # Read the image file and process it with Textract
    with open(path, 'rb') as ima:
        image_bytes = ima.read()
    
    # Convert image_bytes back to a stream for PIL processing
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')

    # Perform text extraction using Textract
    response = client.detect_document_text(Document={'Bytes': image_bytes})

    blocks = response['Blocks']
    width, height = image.size
    print('Detected Document Text')

    # Getting the text and lower right coordinates of each LINE
    text = []
    coord = []
    for block in blocks:
        if block['BlockType'] == 'LINE':
            text.append(block['Text'])
            left = int(block['Geometry']['BoundingBox']['Left'] * width)
            top = int((block['Geometry']['BoundingBox']['Top'] + block['Geometry']['BoundingBox']['Height']) * height)
            coord.append((left, top))

    print(text)
    return len(blocks), text, coord
