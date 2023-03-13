import unittest
from model import OnnxModel, PreProcessing
from PIL import Image


class MyTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.onnx_obj = OnnxModel(model_name="ImageClassifier.onnx")
        self.onnx_obj.load_model()
        self.preprocess = PreProcessing()

    def get_model_output(self, img_path):
        img = Image.open(img_path)
        preprocessed_img = self.preprocess.preprocess_image(img, (224, 224)).unsqueeze(0)
        output = self.onnx_obj.generate_prediction(preprocessed_img)
        return output

    def test_tench_img(self):
        output = self.get_model_output("./n01440764_tench.JPEG")
        self.assertEqual(0, output)

    def test_turtle_img(self):
        output = self.get_model_output("./n01667114_mud_turtle.JPEG")
        self.assertEqual(35, output)


if __name__ == '__main__':
    unittest.main()
