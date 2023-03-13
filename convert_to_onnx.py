import torch.onnx


def convert_pytorch_to_onnx(model, input_tensor, model_name,
                            model_input_names, model_output_names,
                            export_params=True, opset_version=10,
                            do_constant_folding=True):
    """
    convert a pytorch model to onnx model
    param model: model being run
    param input_tensor: model input
    param model_name: where to save the model
    param model_input_names: the model's input names
    param model_output_names: the model's output names
    param export_params: store the trained parameter
    param opset_version: the ONNX version to export the model to
    param do_constant_folding: optimization
    return None
    """

    model.eval()
    torch.onnx.export(model,
                      input_tensor,
                      model_name,
                      export_params=export_params,
                      opset_version=opset_version,
                      do_constant_folding=do_constant_folding,
                      input_names=model_input_names,
                      output_names=model_output_names)
    print('Model converted to ONNX')
