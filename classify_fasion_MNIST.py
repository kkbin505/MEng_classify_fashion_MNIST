# Classifyer


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import tensorflow_model_optimization as tfmot 
from tensorflow.keras.layers import Flatten, Dense

# Define data location
DATA_PATH = 'data' # <-- EDIT THIS PATH
TFLITE_FILE_NAME = 'fashion_mnist_full_int16_qat.tflite'
C_HEADER_NAME = 'model_data_int16.h'

# Step 1: Load the Fashion MNIST Dataset from local
def load_fashion_mnist_data(path, kind='train'):
    """Load Fashion MNIST data from idx3-ubyte files."""
    if kind == 'train':
        labels_filename = 'train-labels-idx1-ubyte'
        images_filename = 'train-images-idx3-ubyte'
    else:
        labels_filename = 't10k-labels-idx1-ubyte'
        images_filename = 't10k-images-idx3-ubyte'

    labels_path = os.path.join(path, labels_filename)
    images_path = os.path.join(path, images_filename)

    with open(labels_path, 'rb') as lbpath:
        lbpath.seek(8)
        labels = np.fromfile(lbpath, dtype=np.uint8)

    with open(images_path, 'rb') as imgpath:
        imgpath.seek(16)
        images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(labels), 28, 28)

    return images, labels



try:
    # Load training data
    train_images, train_labels = load_fashion_mnist_data(DATA_PATH, kind='train')
    # Load test data
    test_images, test_labels = load_fashion_mnist_data(DATA_PATH, kind='t10k')
except FileNotFoundError as e:
    print(f"Error loading file: {e.filename}")
    print(f"Please make sure the data files are located in the '{DATA_PATH}' directory.")
    exit()


# Define class names for the labels
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Step 2: Preprocess the Data, normilize pixel data from 0~200 to 0~1.0
train_images = train_images / 255.0
test_images = test_images / 255.0

# Step 3: Build the Neural Network Model
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Step 4: Compile the Model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Display a summary of the model's architecture
print("Model Summary:")
model.summary()

# Step 5: Train the Model
print("\nStarting model training...")
history = model.fit(train_images, train_labels, epochs=10, validation_split=0.1)
print("Base Model Training Finished.")
test_acc_base = model.evaluate(test_images, test_labels, verbose=0)[1]
print(f'Base Model Accuracy: {test_acc_base:.4f}')

# =========================================================================
# Step 6: QUANTIZATION AWARE TRAINING (QAT)
# =========================================================================

# Step 6: QAT API
quantize_model = tfmot.quantization.keras.quantize_model
qat_model = quantize_model(model)
print("\nModel transformed for Quantization-Aware Training.")

qat_model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

print("\nStarting QAT fine-tuning (4 epochs)...")
qat_model.fit(train_images, train_labels, epochs=4, validation_split=0.1)
print("QAT Fine-Tuning Finished.")

test_acc_qat = qat_model.evaluate(test_images, test_labels, verbose=0)[1]
print(f'QAT Model Accuracy after fine-tuning: {test_acc_qat:.4f}')
print(f"QAT successfully recovered accuracy: {test_acc_base - test_acc_qat:.4f} drop.")

# =========================================================================
# Step 7: CONVERSION TO FULL INT16 TFLITE (Final Deployment Model)
# =========================================================================

# Step 7: Make Predictions
def representative_data_gen():
    for input_value in tf.data.Dataset.from_tensor_slices(train_images).batch(1).take(100):
        yield [input_value]

converter = tf.lite.TFLiteConverter.from_keras_model(qat_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# 7.2 convert to int16
converter.representative_dataset = representative_data_gen
converter.target_spec.supported_types = [tf.int16] 
converter.inference_input_type = tf.int16
converter.inference_output_type = tf.int16

tflite_model_full_int16_qat = converter.convert()
print("\n✅ TFLite Full INT16 Model Conversion Successful.")


# =========================================================================
# Step 8: CONVERT AND QUANTIZE MODEL FOR ESP32-S3 (TFLite Micro)
# We use Hybrid Quantization (INT8 interface, aiming for better internal precision)
# =========================================================================
# 1. 保存 TFLite 文件
with open(TFLITE_FILE_NAME, 'wb') as f:
    f.write(tflite_model_full_int16_qat)


# 2. 将 TFLite 文件转换为 C 数组 (替换你的 ESP32 头文件)
tflite_model_binary = tflite_model_full_int16_qat
c_code = (
    f"#ifndef MODEL_DATA_INT16_H\n"
    f"#define MODEL_DATA_INT16_H\n\n"
    f"// Model was generated using Quantization Aware Training (QAT).\n"
    f"const unsigned char g_model_data[] = {{\n  " + 
    ', '.join(f'0x{b:02x}' for b in tflite_model_binary) + 
    "\n};\n"
    f"const int g_model_data_len = {len(tflite_model_binary)};\n\n"
    f"#endif // MODEL_DATA_INT16_H\n"
)

with open(C_HEADER_NAME, 'w') as f:
    f.write(c_code)

print(f"\n🎉 SUCCESS! New QAT Header created: {C_HEADER_NAME}. Now redeploy to ESP32.")