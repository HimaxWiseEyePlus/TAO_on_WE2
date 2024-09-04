import onnx
import argparse

def remove_suffix(input_string, suffix):
    if suffix and input_string.endswith(suffix):
        return input_string[:-len(suffix)]
    return input_string

parser = argparse.ArgumentParser(description='Remove :0 suffix from ONNX model input and output names.')
parser.add_argument('--input', required=True, type=str, help='Path to the input ONNX model')
parser.add_argument('--output', required=True, type=str, help='Path to save the modified ONNX model')
args = parser.parse_args()

onnx_model = onnx.load(args.input)
suffix = ':0'

graph_input_names = [input.name for input in onnx_model.graph.input]
graph_output_names = [output.name for output in onnx_model.graph.output]

for input in onnx_model.graph.input:
    input.name = remove_suffix(input.name, suffix)

for output in onnx_model.graph.output:
    output.name = remove_suffix(output.name, suffix)

for node in onnx_model.graph.node:
    for i in range(len(node.input)):
        if node.input[i] in graph_input_names:
            node.input[i] = remove_suffix(node.input[i], suffix)

    for i in range(len(node.output)):
        if node.output[i] in graph_output_names:
            node.output[i] = remove_suffix(node.output[i], suffix)

onnx.save(onnx_model, args.output)
