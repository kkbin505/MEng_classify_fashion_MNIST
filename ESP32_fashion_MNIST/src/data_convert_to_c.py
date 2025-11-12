import numpy as np
import os

# 确保文件路径正确
TFLITE_MODEL_NAME_HYBRID = 'fashion_mnist_quant_hybrid.tflite'
C_FILE_NAME = 'model_data.h'

# Load the binary TFLite file
with open(TFLITE_MODEL_NAME_HYBRID, 'rb') as f:
    tflite_model_binary = f.read()

# Generate C array code
c_code = (
    '#ifndef MODEL_DATA_H\n'
    '#define MODEL_DATA_H\n\n'
    '// This array contains the model data for TFLite Micro\n'
    'const unsigned char g_model_data[] = {\n  ' + 
    ', '.join(f'0x{b:02x}' for b in tflite_model_binary) + 
    '\n};\n'
    f'const int g_model_data_len = {len(tflite_model_binary)};\n\n'
    '#endif // MODEL_DATA_H\n'
)

# Save the C header file
with open(C_FILE_NAME, 'w') as f:
    f.write(c_code)

print(f"Successfully converted TFLite model to C header: {C_FILE_NAME}")