from transformers import pipeline
import torch
import io
import torch
import requests
import numpy as np
import onnxruntime
from PIL import Image
from model import  Model , Preprocessor
from torchvision import transforms
import base64

# Init is ran on server startup
# Load your model to GPU as a global variable here using the variable name "model"
def init():
    global model

    model_file = "mtailormodel.onnx"
    if torch.cuda.is_available():
        device = 0
    else:
        device = -1
    model = onnxruntime.InferenceSession(model_file, providers=['CUDAExecutionProvider'] if device == 0 else [])

# Inference is ran for every server call
# Reference your preloaded global model variable here.
def inference(model_inputs):
    global model
    # Convert the image data to a PIL Image object
    image_bytes = model_inputs.get('input',None)
    if image_bytes == None:
        return {'message': "No input provided"}
    image_bytes = base64.b64decode(image_bytes.encode('utf-8'))
    image = Image.open(io.BytesIO(image_bytes))
    
    # Apply the image transform and convert to a numpy array
    if image.mode != 'RGB':
        image = image.convert('RGB')

    resize = transforms.Resize((224, 224))
    crop = transforms.CenterCrop((224, 224))
    to_tensor = transforms.ToTensor()
    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    image = resize(image)
    image = crop(image)
    image = to_tensor(image)
    image = normalize(image)
    

    # Use the ONNX model to make a prediction
    input_name = model.get_inputs()[0].name
    #output_name = model.get_outputs()[0].name
    output = model.run(None, {input_name: image})[0]
    predicted_class = np.argmax(output)

    # Convert the prediction to a JSON response
    #response = {"class": str(predicted_class)}

    return predicted_class

