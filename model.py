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


class PreProcessing():

    def preprocess_image(self, img, img_size):
        resize = self.resize_img(img_size)
        crop = self.center_crop_img(img_size)
        to_tensor = self.convert_img_to_tensor()
        normalize = self.normalize_img()
        resized_image = resize(img)
        cropped_image = crop(resized_image)
        img_tensor = to_tensor(cropped_image)
        normalized_img = normalize(img_tensor)
        return normalized_img

    def resize_img(self, img_size):
        return transforms.Resize(img_size)

    def center_crop_img(self, img_size):
        return transforms.CenterCrop(img_size)

    def convert_img_to_tensor(self):
        return transforms.ToTensor()

    def normalize_img(self):
        return transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    @staticmethod
    def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
