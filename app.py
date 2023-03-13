from PIL import Image
from model import OnnxModel, PreProcessing

# Init is ran on server startup
# Load your model to GPU as a global variable here using the variable name "model"
def init():
    global model

    model = OnnxModel(model_name="ImageClassifier.onnx")
    model.load_model()


# Inference is ran for every server call
# Reference your preloaded global model variable here.
def inference(model_inputs: dict) -> dict:
    global model

    # Parse out your arguments
    img_path = model_inputs.get('img_path', None)
    if img_path == None:
        return {'message': "No img_path provided"}

    preprocess = PreProcessing()
    img = Image.open(img_path)
    preprocessed_img = preprocess.preprocess_image(img, (224, 224)).unsqueeze(0)
    # Run the model
    result = model.generate_prediction(preprocessed_img)

    # Return the results as a dictionary
    return result