from model import OnnxModel, PreProcessing
def download_model():
    onnx_obj = OnnxModel(model_name="ImageClassifier.onnx")
    onnx_obj.load_model()

if __name__ == "__main__":
    download_model()