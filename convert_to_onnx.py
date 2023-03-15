import torch
import onnxruntime as ort
import numpy as np
from PIL import Image
import pytorch_model



# exporting the model
def main():
    model_input = torch.randn(1, 3, 224, 224)
    input = (model_input)

    mtailor_model = pytorch_model.Classifier(pytorch_model.BasicBlock, [2,2,2,2])
    torch.onnx.export(mtailor_model, input, 'sample_model2.onnx')

def classify(image_uri):
    # Load the ONNX model
    sess = ort.InferenceSession('./sample_model2.onnx')

    # Load the image
    img = Image.open('./n01667114_mud_turtle.JPEG')

    # get the input shape
    input_shape = sess.get_inputs()[0].shape

    # Resize the image to match the input size of the model
    input_size = (224, 224)
    # input_size = (input_shape[3], input_shape[2])
    img = img.resize(input_size)

    # Convert the image to a numpy array
    img_array = np.array(img).astype(np.float32)

    # Normalize the image
    img_array /= 255.0
    img_array -= np.array([0.485, 0.456, 0.406])
    img_array /= np.array([0.229, 0.224, 0.225])

    # Add batch dimension to the image array
    img_array = np.expand_dims(img_array, axis=0)

    # Get the input and output names
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name

    # Run inference on the image
    outputs = sess.run([output_name], {input_name: img_array})

    # Get the predicted class index
    pred_class_idx = np.argmax(outputs[0])

    # Print the predicted class
    # print(f"Class index: {pred_class_idx}")
    return str(pred_class_idx)

if __name__ == '__main__':
    main()