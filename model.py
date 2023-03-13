import onnxruntime as ort
from torchvision import transforms
import numpy as np


class OnnxModel():
    def __init__(self, model_name):
        self.model_name = model_name
        self.ort_session = None

    def set_ort_session(self, session):
        self.ort_session = session

    def get_ort_session(self):
        return self.ort_session

    def load_model(self):
        self.set_ort_session(ort.InferenceSession(self.model_name))

    def generate_prediction(self, model_input):
        ort_inputs = {self.ort_session.get_inputs()[0].name: PreProcessing.to_numpy(model_input)}
        ort_outs = self.ort_session.run(None, ort_inputs)
        return np.argmax(ort_outs[0])
