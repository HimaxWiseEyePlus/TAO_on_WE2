import tensorflow as tf
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Remove :0 suffix from ONNX model input and output names.')
parser.add_argument('--input', required=True, type=str, help='Path to the input ONNX model')
parser.add_argument('--output', required=True, type=str, help='Path to save the modified ONNX model')
args = parser.parse_args()

converter = tf.lite.TFLiteConverter.from_saved_model(
    args.input
)

def representative_dataset():
  for _ in range(100):
    yield [
        (np.random.randint(0, 255, [1, 240,320,3])).astype(np.float32)/ 255.0
    ]
    
tflite_model = converter.convert()
converter.optimizations = [
    tf.lite.Optimize.DEFAULT
]

converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS_INT8
]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8
converter.representative_dataset = representative_dataset

tflite_quant_model = converter.convert()

with open(args.output, 'wb') as f:
    f.write(tflite_quant_model)