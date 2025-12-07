import numpy as np
import os

# --- Data paths ---
DATA_PATH = 'data'
IMAGES_FILE = os.path.join(DATA_PATH, 't10k-images-idx3-ubyte')
LABELS_FILE = os.path.join(DATA_PATH, 't10k-labels-idx1-ubyte')

# --- Load IDX images ---
def load_idx_images(path):
    with open(path, 'rb') as f:
        f.seek(16)  # Skip header
        return np.fromfile(f, dtype=np.uint8).reshape(-1, 28, 28)

# --- Load IDX labels ---
def load_idx_labels(path):
    with open(path, 'rb') as f:
        f.seek(8)  # Skip header
        return np.fromfile(f, dtype=np.uint8)

images = load_idx_images(IMAGES_FILE)
labels = load_idx_labels(LABELS_FILE)

# --- Randomly select 30 images ---
np.random.seed(30)  # For reproducibility
indices = np.random.choice(len(images), 30, replace=False)
images_selected = images[indices]
labels_selected = labels[indices]

# --- Generate C header file ---
with open('out_model/esp32_test_data.h', 'w') as f:
    f.write("#pragma once\n\n")
    f.write("// Test images for ESP32 (30 samples, 28x28 pixels)\n")
    f.write("const uint8_t test_images[30][28*28] = {\n")
    for img in images_selected:
        img_flat = img.flatten()
        f.write("  {")
        f.write(",".join(str(p) for p in img_flat))
        f.write("},\n")
    f.write("};\n\n")
    
    f.write("// Corresponding labels for test images\n")
    f.write("const uint8_t test_labels[30] = {")
    f.write(",".join(str(l) for l in labels_selected))
    f.write("};\n")

print("Generated C header file 'esp32_test_data.h'.")
