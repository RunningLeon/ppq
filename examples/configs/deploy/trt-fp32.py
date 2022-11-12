input_height = 1024
input_width = 2048

onnx_config = dict(type='onnx',
                   export_params=True,
                   keep_initializers_as_inputs=False,
                   opset_version=11,
                   save_file='end2end.onnx',
                   input_names=['input'],
                   output_names=['output'],
                   input_shape=(input_width, input_height),
                   optimize=True)
codebase_config = dict(type='mmseg', task='Segmentation')
backend_config = dict(
    type='tensorrt',
    common_config=dict(fp16_mode=False, max_workspace_size=1 << 30),
    model_inputs=[
        dict(input_shapes=dict(
            input=dict(min_shape=[1, 3, input_height, input_width],
                       opt_shape=[1, 3, input_height, input_width],
                       max_shape=[1, 3, input_height, input_width])))
    ])
