# test onnx
from model import Model, Preprocessor


def test_one():
    image_file = "n01667114_mud_turtle.jpeg"
    onnx_file = "mtailormodel.onnx"
    
    p = Preprocessor(image_file)
    pimg = p.preprocess_numpy(p.load_image())
    
    m = Model(pimg, onnx_file)
    assert m.load_and_predict() == ('n01667114_mud_turtle', 35)
    
    
def test_two():
    image_file = "n01440764_tench.jpeg"
    onnx_file = "mtailormodel.onnx"
    
    p = Preprocessor(image_file)
    pimg = p.preprocess_numpy(p.load_image())
    
    m = Model(pimg, onnx_file)
    
    assert m.load_and_predict() == ('n01440764_tench', 0)


