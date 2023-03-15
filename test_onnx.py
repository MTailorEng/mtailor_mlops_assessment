import unittest
import convert_to_onnx


class TestConvert(unittest.TestCase):
    def test_file_classification(self):
        tench_classification = convert_to_onnx.classify('./n01440764_tench.jpeg')
        turtle_classification = convert_to_onnx.classify('./n01667114_mud_turtle.jpeg')

        self.assertEqual(tench_classification, "0")
        self.assertEqual(tench_classification, "35")
        

if __name__ == '__main__':
    unittest.main()
