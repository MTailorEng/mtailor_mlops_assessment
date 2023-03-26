#convert pytorch model to ONNX model
 
import torch
from PIL import Image
from pytorch_model import BasicBlock,Classifier

def convert_to_onnx():
    #load the model
    mtailor = Classifier(BasicBlock, [2, 2, 2, 2])
    #load the weights
    mtailor.load_state_dict(torch.load("./pytorch_model_weights.pth"))
    #set to eval,so it can be in inference mode and not training
    mtailor.eval()

    #dummy input
    img = Image.open("./n01440764_tench.jpeg")
    dummy_input = mtailor.preprocess_numpy(img).unsqueeze(0)

    input_names = ['input']
    output_names = ['output']

    # Export the model
    torch.onnx.export(mtailor, dummy_input, "mtailormodel.onnx", verbose=True, input_names=input_names, output_names=output_names)

    print("Model Converted to ONNX")

if __name__ == "__main__":
    convert_to_onnx()    