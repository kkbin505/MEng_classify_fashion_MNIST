import tensorflow as tf
import numpy as np
import os

# --- Configuration ---
# Coat 4
TARGET_LABEL = 4
IMAGE_SIZE = 28 * 28

# Output file path (This file will be included in your C++ project)
C_HEADER_NAME = 'test_image.h'
VARIABLE_NAME = 'g_test_input_data' # Unique variable name for the test image

# 1. Load the Fashion MNIST test data
try:
    # Use Keras built-in loader for simplicity
    (_, _), (test_images_raw, test_labels) = tf.keras.datasets.fashion_mnist.load_data()
except Exception:
    print("FATAL: Could not load data. Check your TensorFlow/Keras installation.")
    exit()

# 2. Normalize and find the target image
test_images = test_images_raw.astype(np.float32) / 255.0
boot_index = np.where(test_labels == TARGET_LABEL)[0][0]
boot_data_flat = test_images[boot_index].flatten()

# 3. Format the data into a C++ array string
c_array_lines = []
for i in range(0, IMAGE_SIZE, 28):
    # Format 28 elements per line for readability, ensuring 'f' suffix for float
    line = [f"{x:.4f}f" for x in boot_data_flat[i:i+28]]
    c_array_lines.append("    " + ", ".join(line) + ",")

c_array_content = "\n".join(c_array_lines).strip().rstrip(',') + "\n"

# 4. Generate the final C header file content
c_code = (
    f"#ifndef TEST_INPUT_DATA_H\n"
    f"#define TEST_INPUT_DATA_H\n\n"
    f"// --- Fashion MNIST Ankle Boot (Label {TARGET_LABEL}) ---\n"
    f"// Array size: {IMAGE_SIZE} elements (28x28, Float32)\n"
    f"const float {VARIABLE_NAME}[{IMAGE_SIZE}] = {{\n"
    f"{c_array_content}"
    f"}};\n\n"
    f"#endif // TEST_INPUT_DATA_H\n"
)

# 5. Write the content to the file
try:
    with open(C_HEADER_NAME, 'w') as f:
        f.write(c_code)
    print(f"\n🎉 Successfully created header file: {C_HEADER_NAME}")
    print(f"Variable name for use in C++: {VARIABLE_NAME}")
    
except Exception as e:
    print(f"Error writing file: {e}")