# Classifyer

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os

# Define data location
DATA_PATH = 'data' # <-- EDIT THIS PATH

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
print("Model training finished.")

# Step 6: Evaluate the Model's Accuracy
print("\nEvaluating model on test data...")
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print(f'\nTest accuracy: {test_acc:.4f}')

# Step 7: Make Predictions
predictions = model.predict(test_images)

# Helper functions to visualize the predictions
def plot_image(i, predictions_array, true_label, img):
    true_label, img = true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img, cmap=plt.cm.binary)
    predicted_label = np.argmax(predictions_array)
    color = 'blue' if predicted_label == true_label else 'red'
    plt.xlabel(f"{class_names[predicted_label]} {100*np.max(predictions_array):2.0f}% ({class_names[true_label]})",
               color=color)

def plot_value_array(i, predictions_array, true_label):
    true_label = true_label[i]
    plt.grid(False)
    plt.xticks(range(10))
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)
    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')

# Let's visualize the prediction for the first image
print("\nVisualizing a prediction...")
i = 0
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions[i], test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions[i], test_labels)
# plt.show()

# Plot training & validation accuracy values
print("\nPlotting training history...")
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

# Plot training & validation loss values
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

plt.tight_layout()
# plt.show()
model.save("fashion_mnist_model.keras")    # SavedModel格式


# =========================================================================
# Step 8: CONVERT AND QUANTIZE MODEL FOR ESP32-S3 (TFLite Micro)
# We use Hybrid Quantization (INT8 interface, aiming for better internal precision)
# =========================================================================

# 8.1 Define the Converter
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# 8.2 Enable Full Integer Quantization with FLOAT16/INT16 Support

# Enable default optimization (including weight quantization)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# 1. Provide a Representative Dataset (Required for full integer quantization)
# This dataset helps determine the min/max range for activation quantization.
def representative_data_gen():
    # Use 100 samples from the training data for calibration
    # The data must be cast to float32 first, as expected by the generator
    for input_value in tf.data.Dataset.from_tensor_slices(train_images).batch(1).take(100):
        yield [tf.cast(input_value, tf.float32)]

converter.representative_dataset = representative_data_gen

# 2. Set Supported Types (Allowing higher internal precision for better accuracy)
# This tells the converter that using INT16 or FLOAT16 for internal operations is acceptable,
# which can help recover accuracy lost in 8-bit quantization.
converter.target_spec.supported_types = [tf.int16] 

# 3. Force Integer Input/Output (Required for TFLite Micro on ESP32)
# We force INT8 I/O as TFLite's conversion API usually only supports INT8/UINT8 as the full integer interface.
# converter.inference_input_type = tf.int16
# converter.inference_output_type = tf.int16


# 8.3 Perform the Conversion (This creates the optimized TFLite model)
tflite_model_int16 = converter.convert()

# 8.4 Save the Quantized Model
TFLITE_MODEL_NAME_INT16 = 'fashion_mnist_quant_int16.tflite'
with open(TFLITE_MODEL_NAME_INT16, 'wb') as f:
    f.write(tflite_model_int16)

print(f"\nHybrid INT8/INT16 Quantized TFLite model saved to: {TFLITE_MODEL_NAME_INT16}")
print(f"Model size: {len(tflite_model_int16) / 1024:.2f} KB")

# =========================================================================
# Step 9: Evaluate the Quantized Model (Crucial Check)
# =========================================================================

# Evaluate the TFLite model to check for accuracy drop due to quantization.
interpreter = tf.lite.Interpreter(model_content=tflite_model_int16)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()[0]
output_details = interpreter.get_output_details()[0]

# Preprocess test images for INT8 model (Scale from 0.0-1.0 back to 0-255 and cast to int8)
# The input range for the INT8 model is typically [-128, 127] or [0, 255], 
# depending on the zero-point determined by quantization. Here we assume 0-255 scale.
test_images_float32 = test_images.astype(np.float32)

tflite_predictions = []
num_test_samples = 1000 # Increase samples for better confidence
correct_predictions = 0

for i in range(num_test_samples):
    # Reshape and set the input tensor
    input_data = test_images_float32[i:i+1].reshape(1, 28, 28)
    interpreter.set_tensor(input_details['index'], input_data)
    
    # Run inference
    interpreter.invoke()
    
    # Get the output tensor and find the predicted class index
    output = interpreter.get_tensor(output_details['index'])
    predicted_label = np.argmax(output[0])
    
    if predicted_label == test_labels[i]:
        correct_predictions += 1

tflite_accuracy = correct_predictions / num_test_samples
print(f"Quantized TFLite model accuracy (on {num_test_samples} samples): {tflite_accuracy:.4f}")